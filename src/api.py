from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import joblib
import numpy as np
import os

app = FastAPI(
    title='ETA Prediction API',
    description='Predicts NYC taxi trip duration in seconds',
    version='1.0.0'
)

MODEL_NAME = os.getenv('MODEL_NAME', 'XGBoost')
model  = joblib.load(f'models/{MODEL_NAME}_model.pkl')
scaler = joblib.load(f'models/{MODEL_NAME}_scaler.pkl')

class TripRequest(BaseModel):
    passenger_count:             int
    pickup_latitude:             float
    pickup_longitude:            float
    dropoff_latitude:            float
    dropoff_longitude:           float
    hour_of_day:                 int
    day_of_week:                 int
    month:                       int
    is_weekend:                  int
    is_rush_hour:                int
    is_night:                    int
    distance_km:                 float
    direction:                   float
    pickup_distance_from_center: float

    @validator('passenger_count')
    def validate_passengers(cls, v):
        if not 1 <= v <= 6:
            raise ValueError('passenger_count must be 1-6')
        return v

    @validator('distance_km')
    def validate_distance(cls, v):
        if v < 0 or v > 200:
            raise ValueError('distance_km must be 0-200')
        return v

class TripResponse(BaseModel):
    predicted_duration_seconds: float
    predicted_duration_minutes: float
    model_used:                 str

@app.get('/')
def root():
    return {'message': 'ETA Prediction API is running', 'version': '1.0.0'}

@app.get('/health')
def health():
    return {'status': 'healthy', 'model': MODEL_NAME}

@app.post('/predict', response_model=TripResponse)
def predict(trip: TripRequest):
    try:
        features = np.array([[
            trip.passenger_count, trip.distance_km, trip.hour_of_day,
            trip.day_of_week, trip.month, trip.is_weekend,
            trip.is_rush_hour, trip.is_night, trip.direction,
            trip.pickup_distance_from_center, trip.pickup_latitude,
            trip.pickup_longitude, trip.dropoff_latitude, trip.dropoff_longitude
        ]])
        scaled = scaler.transform(features)
        log_pred = model.predict(scaled)[0]
        duration_sec = float(np.expm1(log_pred))
        return TripResponse(
            predicted_duration_seconds=round(duration_sec, 2),
            predicted_duration_minutes=round(duration_sec/60, 2),
            model_used=MODEL_NAME
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)