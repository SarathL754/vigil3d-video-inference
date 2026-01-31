from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException

from app.infer import predict_video  # <- NEW API
#from app.video_io import quick_video_info

import torch

app = FastAPI(title="RWF-2000 Violence Detection API", version="1.0")

DEVICE = torch.device("cpu")  # keep CPU for EC2
DEFAULT_THRESHOLD = 0.5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = DEFAULT_THRESHOLD):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    allowed_ext = {".mp4", ".avi", ".mov", ".mkv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(list(allowed_ext))}",
        )

    # save upload to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            contents = await file.read()
            tmp.write(contents)

        # run inference
        out = predict_video(tmp_path, device=DEVICE, threshold=threshold)

        return {
            "filename": file.filename,
            **out,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # cleanup temp file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
