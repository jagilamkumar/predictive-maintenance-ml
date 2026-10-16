from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal


# Locate the project root and saved model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "predictive_maintenance.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}. "
        "Train and save the model before starting the API."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Predicts machine failure from machine operating measurements "
        "using a trained machine learning pipeline."
    ),
    version="1.0.0",
)


class MachineInput(BaseModel):
    product_type: Literal["L", "M", "H"] = Field(
        description="Product quality type"
    )
    air_temperature_k: float = Field(
        ge=250, le=350, description="Air temperature in Kelvin"
    )
    process_temperature_k: float = Field(
        ge=250, le=400, description="Process temperature in Kelvin"
    )
    rotational_speed_rpm: int = Field(
        ge=1, le=10000, description="Rotational speed in RPM"
    )
    torque_nm: float = Field(
        ge=0, le=500, description="Torque in Newton-metres"
    )
    tool_wear_min: int = Field(
        ge=0, le=1000, description="Tool wear in minutes"
    )


@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API is running",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(data: MachineInput):
    try:
        # Use the exact feature names expected during model training.
        input_df = pd.DataFrame([{
            "Type": data.product_type,
            "Air temperature [K]": data.air_temperature_k,
            "Process temperature [K]": data.process_temperature_k,
            "Rotational speed [rpm]": data.rotational_speed_rpm,
            "Torque [Nm]": data.torque_nm,
            "Tool wear [min]": data.tool_wear_min,
        }])

        prediction = int(model.predict(input_df)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_df)[0][1])

        return {
            "machine_failure_prediction": prediction,
            "prediction_label": (
                "Failure predicted"
                if prediction == 1
                else "No failure predicted"
            ),
            "failure_probability": probability,
            "note": (
                "This is a machine learning prediction, "
                "not a guarantee of machine condition."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}",
        ) from exc
