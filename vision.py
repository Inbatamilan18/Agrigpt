"""
vision.py — Crop disease detection from a leaf photo
=====================================================
Uses a ready-made ViT model fine-tuned on the PlantVillage dataset
(38 diseases of 14 crops). For the 5-hour MVP we skip training;
the final version fine-tunes YOLOv8s (see README "Final version").
"""

import io
import logging

from PIL import Image

logger = logging.getLogger("agrigpt.vision")

VISION_MODEL = "kimcomehome/plantvillage-vit-leaf-disease"

_processor = None
_model = None


def _load():
    global _processor, _model
    if _model is None:
        from transformers import ViTForImageClassification, ViTImageProcessor

        logger.info("Loading plant-disease ViT model (first use)...")
        _processor = ViTImageProcessor.from_pretrained(VISION_MODEL)
        _model = ViTForImageClassification.from_pretrained(VISION_MODEL)
        _model.eval()
    return _processor, _model


def predict_disease(image_bytes: bytes) -> dict:
    """Return {'disease': str, 'confidence': percent} for a photo."""
    import torch

    proc, model = _load()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Could not read that image file. Please send a JPG or PNG.")

    inputs = proc(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    score, idx = torch.max(probs, dim=-1)
    label = model.config.id2label[idx.item()]
    return {"disease": label, "confidence": round(float(score.item()) * 100, 1)}
