# 🧠 AI vs Real Image Classifier

A complete deep-learning pipeline to detect AI-generated images from real photographs.  
Architecture: **EfficientNetB3** with two-phase fine-tuning, real-world augmentation, and TTA inference.

---

## 📁 Project Structure

```
ai-vs-real-classifier/
├── README.md
├── requirements.txt
├── Dockerfile
├── config.yaml
├── data/
│   ├── README.md
│   └── dataset_link.txt
├── src/
│   ├── data_loader.py    # Dataset loading & splitting
│   ├── preprocess.py     # Image preprocessing & augmentation
│   ├── features.py       # Feature extraction helpers
│   ├── train.py          # Model training (Phase 1 + Phase 2)
│   ├── evaluate.py       # Metrics, confusion matrix, ROC
│   ├── predict.py        # Single-image inference with TTA
│   └── utils.py          # Logging, seeding, helpers
├── pipeline/
│   └── pipeline.py       # End-to-end orchestration
├── models/
│   ├── model_v1.keras    # Trained model (after training)
│   └── class_names.json  # Class config & thresholds
├── app/
│   ├── app.py            # FastAPI backend
│   └── routes.py         # API route definitions
├── frontend/
│   ├── index.html        # Upload UI
│   ├── style.css         # Styling
│   └── script.js         # Fetch + display logic
├── logs/
│   └── app.log
└── notebooks/
    └── training.ipynb    # Original Colab notebook
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

See `data/dataset_link.txt` — download from Kaggle and place the zip at:
```
data/ai-generated-images-vs-real-images.zip
```

### 3. Train the Model

```bash
python src/train.py --zip data/ai-generated-images-vs-real-images.zip
```

This runs both training phases and saves:
- `models/model_v1.keras`
- `models/class_names.json`

### 4. Evaluate

```bash
python src/evaluate.py --model models/model_v1.keras --data data/dataset
```

Outputs: accuracy, AUC, confusion matrix, ROC curve → saved to `logs/`.

### 5. Run Inference on a Single Image

```bash
python src/predict.py --image path/to/image.jpg
```

Returns:
```json
{
  "label": "AI",
  "confidence": 0.87,
  "ai_prob": 0.87,
  "real_prob": 0.13
}
```

### 6. Start the Web App

```bash
uvicorn app.app:app --reload --port 8000
```

Open `frontend/index.html` in your browser (or serve via `python -m http.server 3000` from `frontend/`).

---

## 🐳 Docker

```bash
docker build -t ai-vs-real .
docker run -p 8000:8000 ai-vs-real
```

> Deployment details managed separately.

---

## 🏗️ Architecture

| Component | Detail |
|-----------|--------|
| Backbone | EfficientNetB3 (ImageNet weights) |
| Head | GAP → BN → Dense(512) → BN → Dropout(0.5) → Dense(256) → BN → Dropout(0.4) → Sigmoid |
| Phase 1 | Head-only training, base frozen, LR=3e-4, 12 epochs |
| Phase 2 | Top ~50 base layers unfrozen, LR=2e-5, 20 epochs |
| Augmentation | JPEG compression, Gaussian noise, resize artifacts, blur, flips, brightness |
| Inference | TTA × 8 steps, confidence threshold = 0.60 |

---

## 📊 Current Metrics (v1)

| Metric | Value |
|--------|-------|
| Val Accuracy | ~61% |
| Val AUC | ~0.68 |

> v1 performance is intentionally modest — GPU training with full dataset expected to reach 85–90%+ AUC.

---

## 🔧 Configuration

Edit `config.yaml` to change:
- `img_size`, `batch_size`, `epochs`
- `confidence_threshold`
- `tta_steps`
- `dataset_root`, `model_path`
