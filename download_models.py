"""
download_models.py
-------------------
Downloads the heavy model files (transformer weights, ULMFiT model, GloVe
embeddings) from Google Drive and places each one in its correct folder.

These files are excluded from the repository due to size. They are required
only for the dashboard and for retraining; the notebooks run in display mode
without them.

Requires the 'gdown' package (listed in requirements.txt). Install it with:
    pip install gdown

Usage:
    python download_models.py
"""

import os

try:
    import gdown
except ImportError:
    raise SystemExit(
        "The 'gdown' package is required. Install it with:  pip install gdown"
    )

# Project root = the folder this script is in
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Each entry: (Google Drive file ID, destination path relative to project root)
FILES = [
    ("1x08LsscDwFQRkqV_uz1cJIaAPaSpTy1Q", "models/embeddings/glove.twitter.27B.100d.txt"),
    ("1pKcKqleiN--SK49DW9OVXj0hl9uR_Xk_", "models/ulmfit/ulmfit_classifier.pkl"),
    ("193EQRF2qSeb-n8JFxrbNRFL-t-3kpNNj", "models/transformers/deberta/model.safetensors"),
    ("1inGvDQgnxXrnhhPgVJfjlUGDSRmSqlla", "models/transformers/distilbert/model.safetensors"),
    ("1hdH-BSvy70ByTFkm7VFRxeoOgeIY5X70", "models/transformers/roberta/model.safetensors"),
]


def main():
    print("=" * 60)
    print("Downloading heavy model files from Google Drive")
    print("=" * 60)

    for file_id, rel_dest in FILES:
        dest = os.path.join(PROJECT_ROOT, rel_dest)
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        if os.path.exists(dest):
            print(f"[skip] {rel_dest} already exists.")
            continue

        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"\n[get ] {rel_dest}")
        try:
            gdown.download(url, dest, quiet=False)
        except Exception as e:
            print(f"  ERROR downloading {rel_dest}: {e}")
            print("  Check your internet connection and that the file is shared as "
                  "'Anyone with the link'.")

    print("\nAll downloads attempted. The models are now in place.")


if __name__ == "__main__":
    main()