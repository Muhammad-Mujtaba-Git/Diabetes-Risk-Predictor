# 🩺 Pakistan Diabetes Risk Predictor

> A production-grade medical screening microservice — not just a notebook.

Built with **PyTorch + FastAPI + Docker**, trained on real Pakistani hospital data from DHQTH Sahiwal. Predicts diabetes risk from 12 physical metrics with full SHAP explainability.

> ⚠️ Screening tool only. Not a medical diagnosis. Always consult a doctor.

---

## 🔗 Links

| | |
|---|---|
| 🌐 Live Web App | [Try it here](INSERT_STREAMLIT_URL) |
| ⚙️ API Docs | [FastAPI Swagger](INSERT_HUGGINGFACE_URL/docs) |
| 🐳 Docker Image | [HuggingFace Space](INSERT_HUGGINGFACE_URL) |

---

## 🧠 The Data Science Story

### Problem 1 — Data Leakage (Solved)
Raw dataset had 19 features including A1c, nephropathy, and vision loss.
These are **consequences** of diabetes, not causes.
Stripped down to **12 accessible early-warning features** — things a person can measure without being already diagnosed.

### Problem 2 — Sampling Bias (Caught & Removed)
Age had **0.77 correlation** with outcome. Looked useful. Was a trap.

| Group | Source | Avg Age |
|-------|--------|---------|
| Diabetic | Hospital patients | 46 yrs |
| Non-Diabetic | University campus | 25 yrs |

Model would have learned "student vs hospital patient" not diabetes risk.
**Age was dropped entirely.**

### Problem 3 — Confounding Variables (Documented via SHAP)
SHAP showed exercise increasing diabetes risk — medically backwards.
Hospital diabetic patients were **prescribed** exercise because of their diagnosis.
Model learned correlation not causation. Documented honestly, not hidden.

---

## 📊 Final Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 93.96% |
| Train Accuracy | 95.86% |
| **ROC-AUC** | **0.97** |
| Overfitting Gap | 1.9% ✅ |
| LR Baseline | 91.21% |

---

## 🏗️ Architecture

```
Streamlit Frontend
      ↓
FastAPI + Pydantic Validation
      ↓
StandardScaler Preprocessing
      ↓
PyTorch FNN (12 → 24 → 16 → 1)
      ↓
BCEWithLogitsLoss → Sigmoid → Risk Score
      ↓
SHAP Explainability per patient
```

---

## 📁 Project Structure

```
├── model/
│   ├── diabetes_model.pth     # Trained PyTorch weights
│   ├── model.py               # Neural network architecture
│   └── preprocessor.pkl       # Fitted StandardScaler
├── schemas/
│   └── userinput.py           # Pydantic request validation
├── Dockerfile                 # Non-root security optimized container
├── main.py                    # FastAPI endpoints
├── requirements.txt
└── streamlit_app.py           # Frontend UI
```

---

## 🚀 Run Locally

### With Docker
```bash
git clone https://github.com/Muhammad-Mujtaba-Git/Diabetes-Risk-Predictor.git
cd diabetes-risk-predictor
docker build -t diabetes-api .
docker run -p 7860:7860 diabetes-api
```
API live at `http://localhost:7860`
Docs at `http://localhost:7860/docs`

### Without Docker
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## ⚙️ API Endpoints

### `POST /predict`
```json
// Request
{
    "gender": "Male",
    "region": "Urban",
    "weight": 85.0,
    "height": 175.0,
    "waist": 38.0,
    "systolic": 130,
    "diastolic": 85,
    "family_history": "Yes",
    "thirst": "No",
    "urination": "No",
    "hdl": 45.0,
    "exercise_hours": 0.5
}

// Response
{
    "status": "success",
    "data": {
        "prediction": 1,
        "probability": "73.24%",
        "diagnosis": "Diabetic (High Risk)",
        "details": {
            "computed_bmi": 27.76,
            "features_used": 12
        }
    }
}
```

### `POST /explain`
Returns SHAP values per feature explaining the prediction.

### `GET /ping`
Health check endpoint.

---

## 🐳 Docker

Security optimized — runs on non-root user (UID 1000):

```dockerfile
FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
```

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | PyTorch 2.0 |
| Explainability | SHAP |
| API | FastAPI + Pydantic |
| Containerization | Docker |
| Frontend | Streamlit |
| Hosting | HuggingFace Spaces |
| Uptime | GitHub Actions |

---

## 📊 Dataset

**Pakistani Diabetes Dataset** by Dr. Muhammad Shoaib & Aysha Qamar
- 486 diabetic records — DHQTH Sahiwal
- 426 non-diabetic records — COMSATS University Sahiwal
- Supervised by Dr. Sarfaraz Ahmad Khan, Diabetes Specialist

---

## 👨‍💻 Author

**Muhammad Mujtaba**
Self-taught ML Engineer | Python • PyTorch • FastAPI • Docker

[LinkedIn](INSERT_LINKEDIN_URL)

---

## ⚕️ Disclaimer

For educational and screening purposes only. Not a substitute for professional medical advice. Always consult a qualified doctor.
