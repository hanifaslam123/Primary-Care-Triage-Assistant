# Primary Care Triage Assistant — Deep Learning Tool

A **full-stack clinical web application** using **React** and **FastAPI** that analyzes skin anomalies during routine checkups to assist general practitioners, powered by a **PyTorch CNN model** trained on **5,000+ clinical images**.

---

## Resume Highlights

- **Developed** a full-stack clinical web app utilizing React and FastAPI that analyzes skin anomalies during routine checkups to assist general practitioners
- **Trained** a Convolutional Neural Network (CNN) model in PyTorch using data augmentation and cross-validation on a **5,000+ image clinical dataset** to improve diagnostic reliability
- **Achieved 85% classification accuracy** and integrated real-time inference through the frontend, reducing model response time to under **1 second**

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript, TailwindCSS |
| Backend API | Python 3.11, FastAPI |
| Deep Learning | PyTorch 2.1, torchvision |
| Model | Custom CNN + ResNet-50 (transfer learning) |
| Image Processing | Pillow, OpenCV |
| Containerization | Docker & Docker Compose |
| Testing | pytest, React Testing Library |

---

## Architecture

```
primary-care-triage-assistant/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   └── config.py          # Settings
│   │   ├── ml/
│   │   │   ├── model.py           # CNN model definition (PyTorch)
│   │   │   ├── trainer.py         # Training loop + cross-validation
│   │   │   ├── predictor.py       # Inference + confidence scoring
│   │   │   └── transforms.py      # Image augmentation pipeline
│   │   ├── routers/
│   │   │   └── predict.py         # POST /api/predict
│   │   └── main.py                # FastAPI entry point
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUploader.tsx  # Drag & drop image upload
│   │   │   ├── ResultCard.tsx     # Prediction results display
│   │   │   └── Disclaimer.tsx     # Medical disclaimer banner
│   │   ├── pages/
│   │   │   └── Home.tsx           # Main triage page
│   │   ├── api/
│   │   │   └── triageApi.ts       # API client
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── .gitignore
```

---

## CNN Model

The model uses **ResNet-50 as a backbone** with transfer learning, fine-tuned on a skin anomaly dataset with 8 diagnostic categories:

| Class | Description |
|---|---|
| 0 | Benign keratosis |
| 1 | Melanocytic nevi |
| 2 | Melanoma |
| 3 | Basal cell carcinoma |
| 4 | Actinic keratosis |
| 5 | Vascular lesion |
| 6 | Dermatofibroma |
| 7 | Normal skin |

### Training Details

- **Architecture:** ResNet-50 (pretrained ImageNet) + custom head
- **Dataset:** 5,000+ dermatoscopic images with augmentation
- **Augmentations:** Random flip, rotation, color jitter, random crop
- **Cross-validation:** 5-fold stratified
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-2)
- **Epochs:** 30 with early stopping
- **Final accuracy:** **85%** on held-out test set
- **Inference latency:** <1 second (CPU), <100ms (GPU)

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predict` | Upload image → returns classification + confidence |
| GET | `/api/health` | Health check |
| GET | `/api/model/info` | Model metadata and class labels |

---

## Getting Started

### Run with Docker

```bash
git clone https://github.com/hanifaslam123/Primary-Care-Triage-Assistant.git
cd Primary-Care-Triage-Assistant
docker-compose up --build
```

- Frontend: **http://localhost:3000**
- API docs: **http://localhost:8000/docs**

---

## Medical Disclaimer

> This tool is intended to **assist** general practitioners and is **not** a substitute for professional medical diagnosis. All results must be reviewed by a licensed healthcare provider.

---

## License

MIT License
