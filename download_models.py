"""
download_models.py — one-time model downloader
==============================================
Run BEFORE starting the server, so models are already cached:

    python download_models.py               # embedding model + disease ViT (~1 GB)
    python download_models.py --with-nllb   # also Hindi translator (+2.5 GB)
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
VISION_MODEL = "kimcomehome/plantvillage-vit-leaf-disease"

print(f"[1/3] Downloading embedding model: {EMBEDDING_MODEL}")
from sentence_transformers import SentenceTransformer

SentenceTransformer(EMBEDDING_MODEL)
print("      done.")

print(f"[2/3] Downloading plant-disease ViT: {VISION_MODEL}")
from transformers import ViTForImageClassification, ViTImageProcessor

ViTImageProcessor.from_pretrained(VISION_MODEL)
ViTForImageClassification.from_pretrained(VISION_MODEL)
print("      done.")

if "--with-nllb" in sys.argv:
    print("[3/3] Downloading NLLB-200-distilled-600M (Hindi support, ~2.5 GB)")
    from huggingface_hub import snapshot_download

    snapshot_download("facebook/nllb-200-distilled-600M")
    print("      done.")
else:
    print("[3/3] Skipped NLLB translator. If you want to demo Hindi, re-run with: --with-nllb")

print("\nAll models ready. Now start the server:")
print("    uvicorn main:app --port 8000")
