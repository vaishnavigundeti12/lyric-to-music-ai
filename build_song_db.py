"""
build_song_db.py — Builds a database of song style profiles.

Run once: python build_song_db.py
Creates:  song_style_db.json
"""

import os
import json
import time
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    print("❌ HF_TOKEN not set. Run: $env:HF_TOKEN=\"hf_...\"")
    exit(1)

LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
llm_client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN, provider="auto")


SONGS = [
    {
        "title": "Ordinary",
        "artist": "Alex Warren",
        "lyrics": (
            "We were just ordinary people, living ordinary lives, "
            "till you came along and made me feel like I could touch the sky"
        ),
    },
    {
        "title": "Arcade",
        "artist": "Duncan Laurence",
        "lyrics": (
            "Loving you is a losing game, all I know, all I know, "
            "loving you is a losing game"
        ),
    },
    {
        "title": "7 Years",
        "artist": "Lukas Graham",
        "lyrics": (
            "Once I was seven years old, my mama told me, "
            "go make yourself some friends or you'll be lonely"
        ),
    },
    {
        "title": "Perfect",
        "artist": "Ed Sheeran",
        "lyrics": (
            "I found a love for me, darling just dive right in, follow my lead, "
            "I found a girl beautiful and sweet"
        ),
    },
    {
        "title": "Slut!",
        "artist": "Taylor Swift",
        "lyrics": (
            "And if they call me a slut, you know it might be worth it for once, "
            "and if I'm gonna be drunk, might as well be drunk in love"
        ),
    },
]


STYLE_PROMPT = """You are an expert music director and producer.

Analyze the following song excerpt and recommend its musical style.

SONG: {title} by {artist}
LYRICS EXCERPT: {lyrics}

Respond with ONLY a valid JSON object — no markdown fences, no commentary.

{{
  "genre": "primary genre",
  "mood": "2-3 word mood descriptor",
  "tempo_bpm": integer between 60 and 160,
  "key": "musical key",
  "instrumentation": ["instrument1", "instrument2", "instrument3", "instrument4"],
  "vocal_style": "description of ideal vocal delivery",
  "production_notes": "brief production direction"
}}

Output JSON only.
"""


def clean_llm_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```", 2)
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip().rstrip("`").strip()


def analyze_song(song: dict) -> dict:
    prompt = STYLE_PROMPT.format(**song)
    try:
        response = llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.5,
        )
        raw = response.choices[0].message.content or ""
        return json.loads(clean_llm_json(raw))
    except Exception as e:
        print(f"  ⚠️  Failed: {e}")
        return None


def main():
    print(f"Building song style database ({len(SONGS)} songs)...\n")
    db = []
    for i, song in enumerate(SONGS, 1):
        print(f"[{i}/{len(SONGS)}] {song['title']} — {song['artist']}")
        style = analyze_song(song)
        if style:
            db.append({
                "title": song["title"],
                "artist": song["artist"],
                "style": style,
            })
            print(f"       ✓ {style.get('genre')} / {style.get('mood')}")
        else:
            print(f"       ✗ Skipped")
        time.sleep(0.5)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "song_style_db.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(db)} profiles to: {out_path}")


if __name__ == "__main__":
    main()