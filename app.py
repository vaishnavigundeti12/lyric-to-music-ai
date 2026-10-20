"""
Intelligent Lyric-to-Music Style Recommendation & Generation
OpenRouter Edition
"""

import os
import json
import re
import textwrap
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf
from openai import OpenAI
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- OpenRouter Configuration ---
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
if not OPENROUTER_KEY:
    print("⚠️  WARNING: OPENROUTER_API_KEY not set.")
    print("   Run: $env:OPENROUTER_API_KEY=\"sk-or-v1-...\"")
else:
    print("✅ OpenRouter API Key detected")

# Initialize OpenRouter Client (OpenAI compatible)
llm_client = None
if OPENROUTER_KEY:
    llm_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
    )
    print("✅ LLM ready: openrouter/free\n")

LLM_MODEL = "openrouter/free"
MUSIC_MODEL = "facebook/musicgen-small"
EMBED_MODEL = "all-MiniLM-L6-v2"
AUDIO_TOKENS = 256


# ============================================================
# INIT — MUSICGEN
# ============================================================

print(f"Loading {MUSIC_MODEL}...")
try:
    music_pipeline = pipeline("text-to-audio", model=MUSIC_MODEL)
    print("✅ MusicGen ready\n")
except Exception as e:
    print(f"❌ MusicGen failed: {e}\n")
    music_pipeline = None


# ============================================================
# INIT — EMBEDDINGS
# ============================================================

print(f"Loading embedding model {EMBED_MODEL}...")
try:
    embedder = SentenceTransformer(EMBED_MODEL)
    print("✅ Embedder ready\n")
except Exception as e:
    print(f"❌ Embedder failed: {e}\n")
    embedder = None


# ============================================================
# LOAD SONG DB
# ============================================================

SONG_DB_PATH = os.path.join(BASE_DIR, "song_style_db.json")
song_db = []
try:
    with open(SONG_DB_PATH, "r", encoding="utf-8") as f:
        song_db = json.load(f)
    print(f"✅ Loaded {len(song_db)} song profiles\n")
except FileNotFoundError:
    print("⚠️  song_style_db.json not found\n")


# ============================================================
# PROMPT INJECTION DETECTION
# ============================================================

INJECTION_PATTERNS = [
    r"ignore (the )?(above|previous|prior) (instructions|prompt)",
    r"disregard (the )?(above|previous|prior)",
    r"you are now",
    r"new instructions",
    r"system:\s*",
    r"</?system>",
    r"forget (everything|all)",
    r"override (the )?instructions",
]


def detect_prompt_injection(text: str):
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return True, f"Matched pattern: '{pattern}'"
    return False, ""


# ============================================================
# CoT PROMPT (5-STEP)
# ============================================================

SYSTEM_MESSAGE = (
    "You are an expert music director and producer. "
    "You always respond with a single valid JSON object after reasoning, "
    "and never include commentary outside the requested format."
)

STYLE_ANALYSIS_PROMPT = """You are an expert music director. Your goal is to translate the user's stylistic request and the emotional content of the lyrics into a structured musical blueprint.

USER'S REQUESTED STYLE:
{user_style}

LYRICS:
{lyrics}

You must show your reasoning step-by-step, explicitly showing how you merge the user's prompt with the lyrics.

STEP 1 — Deconstruct the User's Request:
Analyze the user's requested style. What genre, mood, tempo, and instrumentation do they want? What is the core vibe they are asking for?

STEP 2 — Analyze the Lyrics:
What is the dominant emotion, theme, and imagery of the lyrics? (e.g., sad, angry, nostalgic).

STEP 3 — The Blend (Reconciliation):
How do you merge the User's Request (STEP 1) with the Lyric's Emotion (STEP 2)?
If they conflict (e.g., "sad lyrics" but "upbeat pop"), explain how you resolve this musically (e.g., using a fast tempo but a minor key). If they align, explain how you amplify that emotion.

STEP 4 — Final Musical Parameters:
Based on the blend in STEP 3, commit to the final genre, mood, BPM, key, instrumentation, and vocal style.

Output your reasoning using this exact format, followed by the JSON on the last line:

STEP 1: <your reasoning>
STEP 2: <your reasoning>
STEP 3: <your reasoning>
STEP 4: <your reasoning>
JSON: {{"genre": "...", "mood": "...", "tempo_bpm": 100, "key": "...", "instrumentation": ["...", "..."], "vocal_style": "...", "production_notes": "..."}}
"""

def get_default_style(note: str = "") -> dict:
    return {
        "genre": "Pop", "mood": "emotional and melodic", "tempo_bpm": 100,
        "key": "C Major",
        "instrumentation": ["piano", "acoustic guitar", "soft drums", "bass"],
        "vocal_style": "expressive and warm",
        "production_notes": "balanced mix", "note": note,
    }


def parse_json_response(text: str) -> dict:
    if not text:
        return get_default_style("Empty")

    json_line_match = re.search(r"JSON:\s*(\{.*\})", text, re.DOTALL)
    if json_line_match:
        try:
            return json.loads(json_line_match.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return get_default_style("Parse failed")


# ============================================================
# TERMINAL PRINTING — CoT VISIBILITY (ROBUST & COMPACT)
# ============================================================

def print_cot_to_terminal(raw_output: str, user_style: str, lyrics_snippet: str):
    """Print the full CoT with constrained width and formatted steps."""
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Constrain width to 65 characters for perfect screenshots
    WIDTH = 65
    line = f"{GREEN}{'═' * WIDTH}{RESET}"
    thin_line = f"{GREEN}{'─' * WIDTH}{RESET}"

    print(f"\n{line}")
    print(f"{BOLD}║  🧠 LLM CHAIN-OF-THOUGHT + FINAL OUTPUT{RESET}")
    print(f"{line}")
    
    wrapped_lyrics = textwrap.fill(lyrics_snippet, width=WIDTH-12, initial_indent="  ", subsequent_indent="  ")
    print(f"{BOLD}📝 LYRICS:{RESET}\n{wrapped_lyrics}")
    
    style_text = user_style if user_style else '(none - inferring from lyrics)'
    wrapped_style = textwrap.fill(style_text, width=WIDTH-10, initial_indent="  ", subsequent_indent="  ")
    print(f"{BOLD}🎨 STYLE:{RESET}\n{wrapped_style}")
    print(f"{thin_line}\n")

    # Regex splits the text at "STEP X" or "JSON:" (ignoring markdown asterisks)
    pattern = r'(?i)((?:\*\*|###)?\s*STEP\s*\d+.*?[:：\-]|JSON:)'
    parts = re.split(pattern, raw_output)

    # If no steps were found (fallback), just print the whole text wrapped
    if len(parts) == 1:
        wrapped = textwrap.fill(raw_output, width=WIDTH-2, initial_indent="  ", subsequent_indent="  ")
        print(wrapped)
    else:
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ""

            if header.upper().replace('*', '').startswith("JSON"):
                print(f"\n{YELLOW}{BOLD}🎯 FINAL JSON OUTPUT:{RESET}")
                json_str = content
                # Extract dictionary braces if there's surrounding text
                match = re.search(r'\{.*\}', json_str, re.DOTALL)
                if match:
                    json_str = match.group(0)
                
                try:
                    parsed = json.loads(json_str)
                    formatted = json.dumps(parsed, indent=2)
                    for j_line in formatted.split('\n'):
                        print(f"{YELLOW}{j_line}{RESET}")
                except:
                    wrapped = textwrap.fill(json_str, width=WIDTH-4, initial_indent="  ", subsequent_indent="  ")
                    print(f"{YELLOW}{wrapped}{RESET}")
            else:
                clean_header = header.replace('*', '').replace('#', '').strip()
                print(f"\n{CYAN}{BOLD}▶ {clean_header}{RESET}")
                if content:
                    wrapped = textwrap.fill(content, width=WIDTH-4, initial_indent="  ", subsequent_indent="  ")
                    print(wrapped)

    print(f"\n{line}\n")


# ============================================================
# LYRIC ANALYSIS (OpenRouter)
# ============================================================

def analyze_lyrics(lyrics: str, user_style: str) -> dict:
    if not lyrics.strip() or llm_client is None:
        return get_default_style("LLM unavailable")

    prompt = STYLE_ANALYSIS_PROMPT.format(
        lyrics=lyrics[:2000],
        user_style=user_style.strip() if user_style.strip() else "(none provided)",
    )

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.7,
        )
        raw = response.choices[0].message.content or ""

        # Print full CoT to terminal
        print_cot_to_terminal(raw, user_style, lyrics.strip()[:80])

        return parse_json_response(raw)

    except Exception as e:
        print(f"[LLM error] {e}")
        return get_default_style(f"Error: {str(e)[:80]}")


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_music_prompt(style: dict, lyrics: str, user_style: str) -> str:
    snippet = " ".join(lyrics.strip().split("\n")[:2])[:100]

    if user_style.strip():
        prompt = f"{user_style.strip()}. "
    else:
        prompt = ""

    parts = [
        f"{style.get('genre','Pop')}",
        f"{style.get('mood','melodic')} mood",
        f"{style.get('tempo_bpm',100)} BPM",
        style.get("key", "C Major"),
        f"featuring {', '.join(style.get('instrumentation',[])[:4])}",
        style.get("production_notes", ""),
        "high quality studio production",
    ]
    prompt += ", ".join(x for x in parts if x)

    if snippet:
        prompt += f", evoking: {snippet}"
    return prompt


# ============================================================
# MUSIC GENERATION
# ============================================================

def normalize_audio(audio):
    a = np.asarray(audio)
    a = np.squeeze(a)
    if a.ndim == 0:
        raise ValueError("Empty audio")
    if a.ndim > 2:
        a = a[0]
    peak = np.max(np.abs(a))
    if peak > 0:
        a = a / peak
    return a.astype(np.float32)


def generate_music(prompt: str):
    if music_pipeline is None:
        return None, "❌ MusicGen not loaded"
    try:
        result = music_pipeline(
            prompt,
            forward_params={"max_new_tokens": AUDIO_TOKENS},
        )
        audio = normalize_audio(result["audio"])
        sr = int(result["sampling_rate"])
        out = os.path.join(tempfile.gettempdir(), "generated_music.wav")
        sf.write(out, audio, sr)
        print(f"[MusicGen OK] {len(audio)/sr:.2f}s")
        return out, "✅ Instrumental generated!"
    except Exception as e:
        print(f"[MusicGen error] {e}")
        return None, f"❌ {str(e)[:180]}"


# ============================================================
# RECOMMENDATION
# ============================================================

def style_to_text(style: dict) -> str:
    return " ".join([
        str(style.get("genre", "")),
        str(style.get("mood", "")),
        str(style.get("key", "")),
        " ".join(style.get("instrumentation", [])),
        str(style.get("vocal_style", "")),
        str(style.get("production_notes", "")),
    ]).strip()


def recommend_songs(user_style: dict, top_k: int = 3):
    if not song_db or embedder is None:
        return []
    try:
        user_text = style_to_text(user_style)
        db_texts = [style_to_text(s["style"]) for s in song_db]

        user_vec = embedder.encode([user_text], normalize_embeddings=True)
        db_vecs = embedder.encode(db_texts, normalize_embeddings=True)

        sims = cosine_similarity(user_vec, db_vecs).flatten()
        ranked = sorted(zip(song_db, sims), key=lambda x: x[1], reverse=True)

        out = []
        for s, sim in ranked[:top_k]:
            preview = s.get("preview_audio")
            abs_audio = os.path.join(BASE_DIR, preview) if preview else None
            if abs_audio and not os.path.exists(abs_audio):
                abs_audio = None
            out.append({
                "title": s["title"],
                "artist": s["artist"],
                "similarity": round(float(sim), 3),
                "genre": s["style"].get("genre"),
                "mood": s["style"].get("mood"),
                "tempo_bpm": s["style"].get("tempo_bpm"),
                "audio": abs_audio,
            })
        return out
    except Exception as e:
        print(f"[recommend error] {e}")
        return []


# ============================================================
# LLM-AS-JUDGE (OpenRouter)
# ============================================================

JUDGE_PROMPT = """You are an evaluator. Given the song lyrics, the user's requested style, and the recommended style JSON, rate how well the recommendation respects the user's request while matching the lyrics.

LYRICS:
{lyrics}

USER'S REQUESTED STYLE:
{user_style}

RECOMMENDED STYLE (JSON):
{style_json}

Score 1-10 on:
- mood_accuracy (does the style match the lyrics' emotion?)
- genre_fit (does the style match the user's request?)

Respond with ONLY this JSON:
{{
  "mood_accuracy": integer 1-10,
  "genre_fit": integer 1-10,
  "reason": "one sentence explanation"
}}
"""


def llm_as_judge(lyrics: str, user_style: str, style: dict) -> dict:
    if llm_client is None or not lyrics.strip():
        return {"mood_accuracy": 0, "genre_fit": 0, "reason": "LLM unavailable"}
    
    prompt = JUDGE_PROMPT.format(
        lyrics=lyrics[:1000],
        user_style=user_style.strip() if user_style.strip() else "(none)",
        style_json=json.dumps(style, ensure_ascii=False),
    )
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict evaluator. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.2,
        )
        raw = response.choices[0].message.content or ""
        return parse_json_response(raw)
    except Exception as e:
        print(f"[judge error] {e}")
        return {"mood_accuracy": 0, "genre_fit": 0, "reason": str(e)[:80]}


# ============================================================
# MAIN PIPELINE
# ============================================================

def full_pipeline(lyrics: str, user_style: str, run_judge: bool):
    if not lyrics.strip():
        return ("⚠️ Enter lyrics.", "—", None, "—",
                "### —", "", None,
                "### —", "", None,
                "### —", "", None,
                "")

    suspicious, reason = detect_prompt_injection(lyrics + " " + user_style)
    injection_note = ""
    if suspicious:
        injection_note = f"\n\n⚠️ **Prompt injection pattern detected** ({reason})."

    style_dict = analyze_lyrics(lyrics, user_style)
    style_display = json.dumps(style_dict, indent=2, ensure_ascii=False)

    music_prompt = build_music_prompt(style_dict, lyrics, user_style)
    audio_path, status = generate_music(music_prompt)

    recs = recommend_songs(style_dict, top_k=3)

    titles, details, audios = [], [], []
    for r in recs:
        titles.append(f"### {r['title']} — {r['artist']}")
        details.append(
            f"Similarity: **{r['similarity']}**  |  "
            f"{r['genre']}  |  {r['mood']}  |  {r['tempo_bpm']} BPM"
        )
        audios.append(r["audio"])

    while len(titles) < 3:
        titles.append("### —"); details.append(""); audios.append(None)

    judge_text = ""
    if run_judge:
        j = llm_as_judge(lyrics, user_style, style_dict)
        judge_text = (
            f"**Mood accuracy:** {j.get('mood_accuracy', '?')}/10  \n"
            f"**Genre fit:** {j.get('genre_fit', '?')}/10  \n"
            f"**Reason:** {j.get('reason', '')}"
        )

    return (
        style_display, music_prompt, audio_path,
        status + injection_note,
        titles[0], details[0], audios[0],
        titles[1], details[1], audios[1],
        titles[2], details[2], audios[2],
        judge_text,
    )


# ============================================================
# UI
# ============================================================

def create_interface():
    with gr.Blocks(title="Intelligent Lyric-to-Music Framework") as demo:
        gr.Markdown("""
        # 🎵 Intelligent Lyric-to-Music Style Recommendation & Generation

        **Foundation of AI & Prompt Engineering — Microproject**

        Enter your lyrics **and** the style of music you want. The LLM will reason step-by-step
        (visible in the terminal) and generate the instrumental in your requested style.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                lyrics_input = gr.Textbox(
                    label="📝 Enter Your Lyrics",
                    placeholder="Paste your lyrics here...",
                    lines=10,
                )
                style_input = gr.Textbox(
                    label="🎨 Desired Music Style (your request)",
                    placeholder=(
                        "e.g., slow rock ballad with electric guitar and heavy reverb\n"
                        "e.g., upbeat electronic pop with synthesizers\n"
                        "e.g., acoustic folk with fingerpicked guitar\n"
                        "Leave empty to let the AI infer from lyrics."
                    ),
                    lines=3,
                )
                judge_toggle = gr.Checkbox(
                    label="Run LLM-as-judge evaluation",
                    value=False,
                )
                generate_btn = gr.Button("🎵 Analyze & Generate", variant="primary", size="lg")
                status_box = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=1):
                style_output = gr.Textbox(label="🎯 Style Profile (JSON)", lines=14, interactive=False)
                prompt_output = gr.Textbox(label="🎼 Music Generation Prompt", lines=4, interactive=False)
                audio_output = gr.Audio(label="🔊 Main Generated Instrumental", type="filepath")

        gr.Markdown("---")
        gr.Markdown("## 🎧 Recommended Tunes (similar style, ranked by semantic similarity)")

        with gr.Row():
            with gr.Column():
                t1 = gr.Markdown("### —"); d1 = gr.Markdown(""); a1 = gr.Audio(type="filepath", label="Preview")
            with gr.Column():
                t2 = gr.Markdown("### —"); d2 = gr.Markdown(""); a2 = gr.Audio(type="filepath", label="Preview")
            with gr.Column():
                t3 = gr.Markdown("### —"); d3 = gr.Markdown(""); a3 = gr.Audio(type="filepath", label="Preview")

        gr.Markdown("---")
        gr.Markdown("## 📊 LLM-as-Judge Evaluation")
        judge_output = gr.Markdown("*Enable the checkbox above and regenerate to see evaluation.*")

        generate_btn.click(
            fn=full_pipeline,
            inputs=[lyrics_input, style_input, judge_toggle],
            outputs=[
                style_output, prompt_output, audio_output, status_box,
                t1, d1, a1, t2, d2, a2, t3, d3, a3,
                judge_output,
            ],
        )

        gr.Examples(
            examples=[
                [
                    "Walking through the city lights\nShadows dancing in the night\nEvery step a memory\nOf what we used to be",
                    "slow indie folk ballad with acoustic guitar and soft piano",
                ],
                [
                    "We rise up from the ashes\nBurning brighter than the sun\nNothing's gonna stop us now\nWe've only just begun",
                    "energetic arena rock with electric guitars and driving drums",
                ],
                [
                    "Rain falling on my window\nAnother lonely night\nI still hear your voice\nWhispering goodbye",
                    "melancholic piano ballad with strings",
                ],
            ],
            inputs=[lyrics_input, style_input],
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)