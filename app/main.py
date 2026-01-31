from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException

from app.video_io import load_clip_tensor, quick_video_info
from app.infer import ViolencePredictor, InferenceConfig

app = FastAPI(title="RWF-2000 Violence Detection API", version="1.0")

# Load model ONCE at startup (kept in memory)
predictor = ViolencePredictor(InferenceConfig(device="cpu"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Basic file validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    # Optional: allow only common video formats
    allowed_ext = {".mp4", ".avi", ".mov", ".mkv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(list(allowed_ext))}",
        )

    # Save uploaded file to a temp location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            contents = await file.read()
            tmp.write(contents)

        # Decode + preprocess -> clip tensor
        clip = load_clip_tensor(tmp_path)

        # Predict
        pred = predictor.predict_clip(clip)

        # Some debug metadata (optional but helpful)
        info = quick_video_info(tmp_path)

        return {
            "filename": file.filename,
            "video_info": info,
            **pred,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # cleanup
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
