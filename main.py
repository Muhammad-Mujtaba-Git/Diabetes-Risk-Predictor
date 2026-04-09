from fastapi import FastAPI ,HTTPException
from schemas.userinput import UserInput
from model.model import NNModel
import torch
import pickle
import os
import pandas as pd

import shap
import numpy as np

from datetime import datetime

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'diabetes_model.pth')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'model', 'preprocessor.pkl')
try:
    
    wrapper = NNModel() 
    model = wrapper.get_model()
  
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
  
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    print("Preprocessor loaded using Pickle.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    raise RuntimeError(f"Startup failed: {e}") 

@app.get("/",status_code =200)
def hello():
    """Return dictionary with greeting message."""
    return {"message":"Hello world!"}

@app.post("/predict", status_code=200)
def predict(user_input: UserInput):
    try:
        features_list = user_input.to_model_input()
        columns = [
            'Gender', 'Rgn', 'wt', 'BMI', 'wst', 'sys', 'dia', 
            'his', 'dipsia', 'uria', 'HDL', 'Exr_hours'
        ]
        features_df = pd.DataFrame([features_list], columns=columns)
        scaled_data = preprocessor.transform(features_df)
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
        with torch.no_grad():
            logits = model(input_tensor)
           
            probability = torch.sigmoid(logits).item()

        # 5. Set the threshold (Standard is 0.5)
        prediction = 1 if probability >= 0.5 else 0
        return {
            "status": "success",
            "data": {
                "prediction": prediction,
                "probability": f"{probability * 100:.2f}%",
                "diagnosis": "Diabetic (High Risk)" if prediction == 1 else "Non-Diabetic (Low Risk)",
                "details": {
                    "computed_bmi": user_input.bmi,
                    "features_used": 12
                }
            }
        }

    except Exception as e:
      
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction Error: {str(e)}"
        )

@app.post("/explain", status_code=200)
def explain(user_input: UserInput):
    try:
        features_list = user_input.to_model_input()
        columns = [
            'Gender', 'Rgn', 'wt', 'BMI', 'wst', 'sys', 'dia',
            'his', 'dipsia', 'uria', 'HDL', 'Exr_hours'
        ]
        
        # Prepare input
        features_df = pd.DataFrame([features_list], columns=columns)
        scaled_data = preprocessor.transform(features_df)
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32)

        # FIX: Create a 'neutral' background (all zeros) 
        # This represents a "baseline" person for the model to compare against
        background = torch.zeros((10, 12), dtype=torch.float32) 

        explainer = shap.DeepExplainer(model, background)
        sv = explainer.shap_values(input_tensor)
        
        # sv is usually a list for multiple outputs, or a numpy array
        if isinstance(sv, list):
            sv_fixed = sv[0].flatten()
        else:
            sv_fixed = sv.flatten()

        feature_names = [
            'Gender', 'Region', 'Weight', 'BMI', 'Waist',
            'Systolic BP', 'Diastolic BP', 'Family History',
            'Excessive Thirst', 'Frequent Urination', 'HDL', 'Exercise Hours'
        ]

        shap_output = []
        for name, value in zip(feature_names, sv_fixed):
            val = float(value)
            shap_output.append({
                "feature": name,
                "shap_value": round(val, 4),
                "impact": "Increases Risk" if val > 0 else "Decreases Risk"
            })

        # Sort by the biggest absolute impact
        shap_output.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "status": "success",
            "explanation": shap_output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation Error: {str(e)}")
@app.get("/ping", status_code=200)
async def ping():
    """Health check endpoint to keep the server awake."""
    return {"status": "awake", "timestamp": datetime.now().isoformat()}