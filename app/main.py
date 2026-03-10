from fastapi import FastAPI, HTTPException
from app.schema import HeartDiseaseInput
from app.prediction import predict_heart_disease


# Create FastAPI app with metadata
app = FastAPI(
    title="Heart Disease Prediction API",
    description="A Machine Learning API that predicts the 10-year risk of heart disease using Logistic Regression.",
    version="1.0.0"
)


# Root endpoint
@app.get("/", tags=["Home"])
def home():
    return {
        "message": "Welcome to the Heart Disease Prediction API",
        "author": "Aryan Dev",
        "model": "Logistic Regression"
    }


# Health check endpoint
@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "API is running successfully"
    }


# Prediction endpoint
@app.post("/predict", tags=["Prediction"])
def predict(data: HeartDiseaseInput):

    try:
        result = predict_heart_disease(data)

        return {
            "success": True,
            "prediction": result["prediction"],
            "probability": result["probability"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    