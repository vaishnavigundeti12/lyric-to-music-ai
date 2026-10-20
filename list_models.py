"""
list_models.py — Diagnostic tool for Hugging Face Inference Providers

Lists all models your HF account can call via the Inference Providers router.
Useful for picking a working model for the Lyric-to-Music project.

Run with:  python list_models.py
"""

import os
import sys
import requests


def main():
    token = os.environ.get("HF_TOKEN", "").strip()

    if not token:
        print("❌ HF_TOKEN is not set.")
        print("   In PowerShell, run:  $env:HF_TOKEN=\"hf_...\"")
        sys.exit(1)

    print(f"✅ HF_TOKEN detected (length: {len(token)})")
    print("Querying https://router.huggingface.co/v1/models ...\n")

    try:
        r = requests.get(
            "https://router.huggingface.co/v1/models",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        sys.exit(1)

    if r.status_code != 200:
        print(f"❌ HTTP {r.status_code}")
        print(r.text[:500])
        sys.exit(1)

    data = r.json().get("data", [])
    print(f"Total available models: {len(data)}\n")

    # Print all model IDs (one per line)
    for m in data:
        print(m["id"])

    # Bonus: group by rough family
    print("\n" + "=" * 60)
    print("Grouped by family (for picking a model):")
    print("=" * 60)

    families = {
        "Llama (Meta)": [],
        "Qwen": [],
        "DeepSeek": [],
        "Mistral": [],
        "Gemma (Google)": [],
        "Others": [],
    }

    for m in data:
        mid = m["id"].lower()
        if "llama" in mid:
            families["Llama (Meta)"].append(m["id"])
        elif "qwen" in mid:
            families["Qwen"].append(m["id"])
        elif "deepseek" in mid:
            families["DeepSeek"].append(m["id"])
        elif "mistral" in mid or "mixtral" in mid:
            families["Mistral"].append(m["id"])
        elif "gemma" in mid:
            families["Gemma (Google)"].append(m["id"])
        else:
            families["Others"].append(m["id"])

    for family, models in families.items():
        print(f"\n▶ {family}  ({len(models)})")
        for mid in models[:15]:
            print(f"   - {mid}")
        if len(models) > 15:
            print(f"   ... and {len(models) - 15} more")


if __name__ == "__main__":
    main()notep