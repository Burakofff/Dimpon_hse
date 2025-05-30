from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint
from typing import Dict
import pickle
import numpy as np
import os

app = FastAPI(title="Mental Health Classifier API")


class ModelInput(BaseModel):
    age: conint(ge=18, le=100)
    gender: conint(ge=0, le=1)
    years_of_experience: conint(ge=0, le=80)
    work_location: conint(ge=0, le=10)
    hours_worked_per_week: conint(ge=0, le=100)
    number_of_virtual_meetings: conint(ge=0, le=50)
    work_life_balance_rating: conint(ge=1, le=5)
    stress_level: conint(ge=1, le=5)
    access_to_mental_health_resources: conint(ge=0, le=1)
    productivity_change: conint(ge=-10, le=10)
    social_isolation_rating: conint(ge=1, le=5)
    satisfaction_with_remote_work: conint(ge=1, le=5)
    company_support_for_remote_work: conint(ge=1, le=5)
    physical_activity: conint(ge=0, le=30)
    sleep_quality: conint(ge=1, le=5)

    class Config:
        schema_extra = {
            "example": {
                "age": 30,
                "gender": 1,
                "years_of_experience": 5,
                "work_location": 1,
                "hours_worked_per_week": 40,
                "number_of_virtual_meetings": 10,
                "work_life_balance_rating": 3,
                "stress_level": 3,
                "access_to_mental_health_resources": 1,
                "productivity_change": 0,
                "social_isolation_rating": 3,
                "satisfaction_with_remote_work": 4,
                "company_support_for_remote_work": 4,
                "physical_activity": 5,
                "sleep_quality": 4
            }
        }


class PredictionResponse(BaseModel):
    prediction: int
    confidence_score: float




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "model.pkl")


# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at path: {MODEL_PATH}")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {str(e)}")


@app.get("/", status_code=200)
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", 
    response_model=PredictionResponse,
    status_code=200,
    description="Predict mental health status based on work and lifestyle factors",
    responses={
        200: {"description": "Successful prediction"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)
def predict(input_data: ModelInput):
    try:
        # Calculate derived features
        age_exp_interaction = input_data.age * input_data.years_of_experience
        workload = input_data.hours_worked_per_week * input_data.number_of_virtual_meetings
        sleep_stress_ratio = round(input_data.sleep_quality / (input_data.stress_level + 1), 2)
        physical_satisfaction = input_data.physical_activity * input_data.satisfaction_with_remote_work

        sleep_stress_ratio = 0 if np.isinf(sleep_stress_ratio) else sleep_stress_ratio

        features = np.array([[
            input_data.age,
            input_data.gender,
            input_data.years_of_experience,
            input_data.work_location,
            input_data.hours_worked_per_week,
            input_data.number_of_virtual_meetings,
            input_data.work_life_balance_rating,
            input_data.stress_level,
            input_data.access_to_mental_health_resources,
            input_data.productivity_change,
            input_data.social_isolation_rating,
            input_data.satisfaction_with_remote_work,
            input_data.company_support_for_remote_work,
            input_data.physical_activity,
            input_data.sleep_quality,
            age_exp_interaction,
            workload,
            sleep_stress_ratio,
            physical_satisfaction
        ]])

        # Get prediction and probability scores
        prediction = model.predict(features)
        confidence_score = float(max(model.predict_proba(features)[0]))

        return {
            "prediction": int(prediction[0]),
            "confidence_score": round(confidence_score, 3)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
