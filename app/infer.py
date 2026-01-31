from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch

from app.video_io import load_clip_tensor
from src.models.r3d import R3D18Violence  # uses your training model class


MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "r3d18_best.pt"


def _download_from_s3_if_needed() -> None:
    """
    Downloads the model from S3 if MODEL_PATH is missing.
    Requires env vars:
      - MODEL_S3_URI = s3://bucket/key
    And AWS creds via IAM role (EC2) or env.
    """
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return

    s3_uri = os.getenv("MODEL_S3_URI", "").strip()
    if not s3_uri.startswith("s3://"):
        raise RuntimeError(
            "MODEL file missing and MODEL_S3_URI not set. "
            "Set MODEL_S3_URI like s3://bucket/models/r3d18_best.pt"
        )

    # lazy import so local dev works without aws deps unless needed
    import boto3

    # parse s3://bucket/key
    no_scheme = s3_uri[len("s3://") :]
    bucket, key = no_scheme.split("/", 1)

    s3 = boto3.client("s3")
    tmp_path = MODEL_PATH.with_suffix(".pt.tmp")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(tmp_path))
    tmp_path.replace(MODEL_PATH)


def load_model(device: torch.device) -> torch.nn.Module:
    _download_from_s3_if_needed()

    model = R3D18Violence(num_classes=2, pretrained=False)
    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    # Your saved checkpoint likely contains either:
    # - raw state_dict, or
    # - {"model_state": state_dict, ...}
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def predict_video(video_path: str, device: torch.device, threshold: float = 0.5) -> dict:
    model = load_model(device)

    x, info = load_clip_tensor(video_path)  # (1,3,T,H,W)
    x = x.to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    fight_prob = float(probs[1].item())
    pred_class = 1 if fight_prob >= threshold else 0

    return {
        "video_info": info,
        "pred_class": pred_class,
        "pred_label": "Fight" if pred_class == 1 else "NonFight",
        "fight_prob": fight_prob,
        "threshold": threshold,
    }
