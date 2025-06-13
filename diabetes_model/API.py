#!/usr/bin/env python
# coding: utf-8

# In[2]:


from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from fastapi.responses import JSONResponse

try:
    model = joblib.load("diabetes_model/diabetes_model.pkl")
    scaler = joblib.load("diabetes_model/scaler.pkl")
    smoking_encoder = joblib.load("diabetes_model/smoking_encoder.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")


app = FastAPI()
diabetes_router = APIRouter()
# Define input data 
class PatientData(BaseModel):
    age: float
    hypertension: str
    heart_disease: str
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float


@diabetes_router.post("/predict_Diabetes", response_class=JSONResponse)
def predict(data: PatientData):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data.dict()])

       
        df["hypertension"] = df["hypertension"].str.lower().map({"yes": 1, "no": 0})
        df["heart_disease"] = df["heart_disease"].str.lower().map({"yes": 1, "no": 0})

        # Encode smoking history
        df["smoking_history_encoded"] = smoking_encoder.transform(df["smoking_history"])

        
        features = ['age', 'hypertension', 'heart_disease', 'smoking_history_encoded', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        X = df[features]

        
        X_scaled = scaler.transform(X)

        # Predict
        prob = model.predict_proba(X_scaled)[0][1]
        pred_label = "Diabetes" if prob >= 0.5 else "No Diabetes"

        # Risk level
        if prob == 0:
            risk = "No risk detected you’re currently in the safe zone."
        elif 0 < prob < 0.3:
            risk = "Low risk  keep up the healthy lifestyle!"
        elif 0.3 <= prob < 0.6:
            risk = "Moderate risk  consider checking with a doctor and improving habits."
        elif 0.6 <= prob < 0.8:
            risk = "High risk it’s recommended to consult a healthcare professional soon."
        else:
            risk = "Very high risk immediate medical attention is strongly advised."
     
        return {
            "prediction": pred_label,
            "probability": f"{round(prob * 100, 2)}%",
            "risk level": risk
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# In[ ]:




