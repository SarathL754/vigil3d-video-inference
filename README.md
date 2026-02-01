# VigiL3D – Video Violence Detection System (End-to-End Deployment)

VigiL3D is an end-to-end deep learning system for violence detection in videos, built using a 3D CNN trained on the RWF-2000 dataset and deployed as a production-ready ML inference API with a modern web frontend.

The system enables users to upload a video clip and instantly receive:

- A fight / non-fight classification
- A confidence score
- A threshold-controlled decision
- Raw JSON output for debugging and analysis

This project demonstrates the complete lifecycle of an ML system, from model training and evaluation to Dockerized deployment on AWS EC2 and a public frontend hosted on Vercel.

🚀 Live Demo

Frontend (Vercel)  
👉 [Live Frontend Demo](https://vigil-frontend-delta.vercel.app/)

Backend API (AWS EC2 – FastAPI)

- Health check: `/health`
- Inference endpoint: `/predict`

⚠️ The backend is hosted on EC2 (CPU inference). Inference time depends on video length.

---

## 🧠 Model Overview

- Architecture: 3D Convolutional Neural Network (3D CNN – ResNet-18 backbone)  
- Dataset: RWF-2000 (Real-World Fight Dataset)  
- Task: Binary classification  
  - Fight  
  - NonFight  
- Inference Mode: CPU (AWS EC2)

The model processes short video clips and learns spatiotemporal features to identify violent interactions.

---

## 📊 Evaluation & Results

The trained model was evaluated using standard classification metrics.

Metrics Used

- Confusion Matrix
<img width="1280" height="960" alt="confusion_matrix" src="https://github.com/user-attachments/assets/ed9020d9-6183-4e81-9dc1-3fc40c955351" />
- Precision-Recall (PR) Curve
<img width="1280" height="960" alt="pr_curve" src="https://github.com/user-attachments/assets/ad652112-f259-4ce4-bc8e-150cafe3f1e9" />
- Fight Probability Distribution
- Threshold-based decision analysis

Key Observations

- Strong separation between fight and non-fight clips
- Adjustable decision threshold allows precision vs recall trade-off
- High confidence predictions on clear violent events


---

## 🏗️ System Architecture

User Browser  
↓  
Vercel Frontend (React + Vite + Tailwind)  
↓  (/api/* rewrite)  
FastAPI Backend (Docker)  
↓  
3D CNN Model (loaded from S3)  
↓  
Prediction JSON Response

---

## ⚙️ Backend – FastAPI Inference Service

Features

- File upload via multipart/form-data
- Threshold-controlled prediction
- Temporary file handling
- Model loaded from Amazon S3
- CORS enabled for public frontend access
- Swagger UI for testing

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | /health  | Health check |
| POST   | /predict | Run violence detection |

### Sample Response

```json
{
  "filename": "sample.mp4",
  "pred_class": 1,
  "pred_label": "Fight",
  "fight_prob": 0.79,
  "threshold": 0.5
}
```

---

## 🐳 Dockerization

The backend is fully containerized for reproducibility.

Why Docker?

- Consistent runtime across machines
- Easy redeployment after EC2 restarts
- Clear separation of inference environment

Key Files

- Dockerfile
- .dockerignore
- requirements-infer.txt

The container runs:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## ☁️ AWS Deployment

### EC2

- Instance type: CPU-based EC2
- Docker installed on Amazon Linux
- Container exposed on port 8000

### S3

- Model weights stored in S3
- Loaded at runtime using environment variable:

```bash
export MODEL_S3_URI=s3://<bucket>/models/r3d18_best.pt
```

This keeps the Docker image lightweight and production-friendly.

---

## 🌐 Frontend – Vercel (React + Vite)

Features

- Drag-and-drop video upload
- Threshold slider
- Loading & progress states
- Result visualization
- Confidence bar
- Raw JSON collapsible panel
- API connectivity status indicator

Tech Stack

- React + TypeScript
- Vite
- Tailwind CSS
- Lucide Icons
- Deployed on Vercel

### API Integration

The frontend communicates with the backend via:

- `VITE_API_BASE=/api`

Vercel rewrite:

```json
{
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "http://<EC2_PUBLIC_IP>:8000/:path*"
    }
  ]
}
```

This avoids CORS issues and keeps the frontend HTTPS-friendly.

---

## 🔐 CORS & Security

- CORS enabled on FastAPI backend
- API accessed only through Vercel rewrite
- No credentials or secrets exposed in frontend
- Model access controlled via S3 IAM permissions

---

## 🧪 Testing

### Swagger UI

- Available at `/docs`
- Used to validate:
  - File upload
  - Threshold handling
  - Model loading
  - Error messages

### Frontend Testing

Verified:

- Successful uploads
- API connectivity
- Error handling
- Result rendering

---

## 📁 Repository Structure

vigil3d-video-inference/  
│  
├── app/                 # FastAPI app  
├── src/                 # Model & inference logic  
├── scripts/             # Utility scripts  
├── Dockerfile  
├── requirements-infer.txt  
├── .dockerignore  
├── .gitignore  
└── README.md

---

## 🎯 Project Goal

The ultimate goal of VigiL3D is to evolve into a:

Real-time CCTV surveillance system that can detect violent activity before human intervention, automatically trigger alerts, and assist authorities or security teams in responding faster.

This project represents the foundational offline inference pipeline toward that vision.

---

## 🔮 Future Work

- 🚀 Upgrade to more powerful architectures (SlowFast, I3D, Video Transformers)
- 🎥 Real-time streaming inference (RTSP / CCTV feeds)
- ⏱️ Sliding-window temporal detection
- 🔔 Alerting system (SMS / Email / Dashboard)
- 🧠 Multi-class violence categorization
- ⚡ GPU inference & autoscaling
- 📈 Improve accuracy with larger datasets & fine-tuning

---

## 🧑‍💻 Author

Sarath L  
MS in Artificial Intelligence
