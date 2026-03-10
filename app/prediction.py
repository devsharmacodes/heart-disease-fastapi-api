import numpy as np
from app.model_loader import load_model

# load model once
model = load_model()


def predict_heart_disease(data):

    # convert input data to numpy array
    features = np.array([
        [
            data.male,
            data.age,
            data.education,
            data.currentSmoker,
            data.cigsPerDay,
            data.BPMeds,
            data.prevalentStroke,
            data.prevalentHyp,
            data.diabetes,
            data.totChol,
            data.sysBP,
            data.diaBP,
            data.BMI,
            data.heartRate,
            data.glucose
        ]
    ])

    # prediction
    prediction = model.predict(features)

    # probability
    probability = model.predict_proba(features)

    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0][1])
    }