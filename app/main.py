from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # ✅ add this

from app.infer import predict_video
import torch

app = FastAPI(title="RWF-2000 Violence Detection API", version="1.0")

# ✅ CORS (add right after app = FastAPI(...))
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:4173",
    # replace with your actual Vercel domain(s)
    "https://vigil-frontend.vercel.app",
    "https://*.vercel.app",  # optional (see note below)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cpu")
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

    tmp_path = None  # ✅ avoid tmp_path referenced before assignment

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            contents = await file.read()
            tmp.write(contents)

        out = predict_video(tmp_path, device=DEVICE, threshold=threshold)

        return {"filename": file.filename, **out}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
