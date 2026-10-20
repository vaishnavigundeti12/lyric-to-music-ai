"""
build_song_audio.py — Pre-generates a short instrumental preview for each song
in song_style_db.json, using MusicGen.

Run once: python build_song_audio.py
Outputs:  song_previews/<title>_<artist>.wav
"""

import os
import json
import numpy as np
import soundfile as sf
from transformers import pipeline

MUSIC_MODEL = "facebook/musicgen-small"
AUDIO_TOKENS = 256
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "song_style_db.json")
OUT_DIR = os.path.join(BASE_DIR, "song_previews")

os.makedirs(OUT_DIR, exist_ok=True)


def style_to_prompt(style: dict) -> str:
    parts = [
        f"{style.get('genre','Pop')} music",
        f"{style.get('mood','melodic')} atmosphere",
        f"{style.get('tempo_bpm',100)} BPM",
        style.get("key", "C Major"),
        f"featuring {', '.join(style.get('instrumentation',[])[:4])}",
        style.get("production_notes", ""),
        "high quality studio production",
    ]
    return ", ".join(p for p in parts if p)


def normalize(audio):
    a = np.asarray(audio)
    a = np.squeeze(a)
    if a.ndim > 2:
        a = a[0]
    peak = np.max(np.abs(a))
    if peak > 0:
        a = a / peak
    return a.astype(np.float32)


def safe_filename(title: str, artist: str) -> str:
    name = f"{title}_{artist}".replace(" ", "_")
    return "".join(c for c in name if c.isalnum() or c in "_-") + ".wav"


def main():
    if not os.path.exists(DB_PATH):
        print(f"❌ {DB_PATH} not found. Run build_song_db.py first.")
        return

    with open(DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)

    print(f"Loaded {len(db)} songs.")
    print("Loading MusicGen...")
    music_pipeline = pipeline("text-to-audio", model=MUSIC_MODEL)
    print("✅ MusicGen ready\n")

    for i, song in enumerate(db, 1):
        prompt = style_to_prompt(song["style"])
        print(f"[{i}/{len(db)}] {song['title']} — {song['artist']}")
        try:
            result = music_pipeline(
                prompt,
                forward_params={"max_new_tokens": AUDIO_TOKENS},
            )
            audio = normalize(result["audio"])
            sr = int(result["sampling_rate"])
            fname = safe_filename(song["title"], song["artist"])
            out_path = os.path.join(OUT_DIR, fname)
            sf.write(out_path, audio, sr)
            song["preview_audio"] = os.path.join("song_previews", fname)
            print(f"      ✓ saved {fname} ({len(audio)/sr:.2f}s)\n")
        except Exception as e:
            print(f"      ✗ failed: {e}\n")
            song["preview_audio"] = None

    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Previews saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()