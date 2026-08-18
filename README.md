## Retraining Strategy

### Trigger Conditions (any ONE of these triggers retraining):
1. MAE degrades more than 15% vs baseline (automated check every 24 hours)
2. Data drift detected on more than 30% of features (Evidently report)
3. More than 10,000 new trip records available since last training
4. Scheduled: every 2 weeks regardless of performance

### Retraining Steps:
1. Pull latest trip data from source system
2. Run validation pipeline (src/validate.py)
3. Run feature engineering (src/features.py)
4. Append to training set (sliding 90-day window)
5. Run training script (src/train.py) — MLflow auto-tracks new experiment
6. Compare new model vs current model on holdout set
7. If new MAE < current MAE: rebuild Docker image and redeploy
8. Update monitoring baseline to new model performance numbers

# ETA Prediction — ML Engineering Mini-Project
Course: PCAM ZC412 | Flavor A | BITS Pilani

## Architecture
Raw CSV → validate.py → features.py → train.py (MLflow) → XGBoost model
→ api.py (FastAPI) → Docker container → Evidently monitoring → retraining

## Setup
git clone https://github.com/yourusername/ml-eta-prediction.git
cd ml-eta-prediction
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

## Run Week 1 (Data Pipeline)
python src/features.py

## Run Week 2 (Training + MLflow)
python src/train.py
mlflow ui  # open http://localhost:5000

## Run Week 3 (API)
python src/api.py  # open http://localhost:8000/docs
# OR with Docker:
docker build -t eta-predictor . && docker run -p 8000:8000 eta-predictor

## Run Week 4 (Monitoring)
python monitoring/monitor.py
# Opens monitoring/drift_report.html