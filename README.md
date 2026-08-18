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