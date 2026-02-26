from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Create FastAPI app
app = FastAPI(
    title="AI Disease Prediction API",
    description="Predict diseases based on symptoms using machine learning",
    version="1.0.0"
)

# Load trained model and symptoms
try:
    model = joblib.load('disease_model.pkl')
    symptoms_list = joblib.load('symptoms_list.pkl')
    try:
        label_encoder = joblib.load('label_encoder.pkl')
        print("Model, symptoms, and label encoder loaded successfully")
    except FileNotFoundError:
        label_encoder = None
        print("Model and symptoms loaded, but label encoder not found")
except FileNotFoundError:
    print("Model files not found. Please run train_model.py first.")
    model = None
    symptoms_list = []

# Disease information database
disease_info = {
    "Common Cold": {
        "description": "A viral infection of the upper respiratory tract.",
        "precautions": ["Rest adequately", "Drink plenty of fluids", "Use saline nasal drops", "Avoid close contact with others"]
    },
    "Influenza": {
        "description": "A contagious respiratory illness caused by influenza viruses.",
        "precautions": ["Get annual flu vaccine", "Rest and stay hydrated", "Take antiviral medications if prescribed", "Isolate to prevent spreading"]
    },
    "COVID-19": {
        "description": "A respiratory disease caused by the SARS-CoV-2 virus.",
        "precautions": ["Get vaccinated", "Wear mask in public", "Maintain social distancing", "Isolate if symptomatic"]
    },
    "Migraine": {
        "description": "A neurological condition characterized by intense headache.",
        "precautions": ["Identify and avoid triggers", "Maintain regular sleep schedule", "Stay hydrated", "Practice stress management"]
    },
    "Hypertension": {
        "description": "High blood pressure that can lead to serious health problems.",
        "precautions": ["Monitor blood pressure regularly", "Reduce sodium intake", "Exercise regularly", "Take prescribed medications"]
    },
    "Diabetes": {
        "description": "A group of metabolic disorders characterized by high blood sugar.",
        "precautions": ["Monitor blood glucose levels", "Follow diabetic diet", "Take prescribed medications", "Regular exercise"]
    },
    "Asthma": {
        "description": "A chronic inflammatory disease of the airways.",
        "precautions": ["Avoid triggers", "Use inhaler as prescribed", "Monitor breathing", "Emergency action plan"]
    },
    "Gastroenteritis": {
        "description": "Inflammation of the stomach and intestines.",
        "precautions": ["Stay hydrated", "BRAT diet (bananas, rice, applesauce, toast)", "Rest", "Avoid dairy and fatty foods"]
    },
    "Arthritis": {
        "description": "Inflammation of one or more joints.",
        "precautions": ["Regular low-impact exercise", "Maintain healthy weight", "Hot/cold therapy", "Anti-inflammatory medications"]
    },
    "Depression": {
        "description": "A mood disorder causing persistent feelings of sadness.",
        "precautions": ["Seek professional help", "Regular exercise", "Maintain social connections", "Follow treatment plan"]
    },
    "Anxiety Disorder": {
        "description": "Excessive worry and fear that interferes with daily activities.",
        "precautions": ["Practice relaxation techniques", "Regular exercise", "Limit caffeine", "Professional counseling"]
    },
    "Allergic Reaction": {
        "description": "Immune system response to allergens.",
        "precautions": ["Identify and avoid allergens", "Take antihistamines", "Carry epinephrine auto-injector", "Emergency medical attention"]
    },
    "Pneumonia": {
        "description": "Infection that inflames air sacs in one or both lungs.",
        "precautions": ["Complete antibiotic course", "Rest and hydration", "Vaccination (pneumococcal)", "Follow-up care"]
    },
    "Bronchitis": {
        "description": "Inflammation of the bronchial tubes.",
        "precautions": ["Stay hydrated", "Use humidifier", "Avoid smoke", "Rest"]
    },
    "Sinusitis": {
        "description": "Inflammation of the tissue lining the sinuses.",
        "precautions": ["Saline nasal irrigation", "Steam inhalation", "Decongestants", "Antibiotics if bacterial"]
    },
    "UTI": {
        "description": "Infection in any part of the urinary system.",
        "precautions": ["Drink plenty of water", "Cranberry juice", "Complete antibiotic course", "Good hygiene"]
    },
    "GERD": {
        "description": "Chronic digestive disease with stomach acid flowing back.",
        "precautions": ["Avoid trigger foods", "Elevate head while sleeping", "Small frequent meals", "Antacids or prescribed medication"]
    },
    "Anemia": {
        "description": "Condition with insufficient healthy red blood cells.",
        "precautions": ["Iron-rich diet", "Vitamin supplements", "Treat underlying cause", "Regular monitoring"]
    },
    "Thyroid Disorder": {
        "description": "Disorder of the thyroid gland affecting metabolism.",
        "precautions": ["Regular thyroid function tests", "Medication compliance", "Balanced diet", "Stress management"]
    },
    "Food Poisoning": {
        "description": "Illness caused by eating contaminated food.",
        "precautions": ["Stay hydrated", "BRAT diet", "Rest", "Seek medical attention if severe"]
    }
}

# Request model
class SymptomRequest(BaseModel):
    symptoms: List[str]
    age: Optional[int] = 25
    gender: Optional[str] = "male"
    severity: Optional[int] = 3  # 1-5 scale

# Response model
class DiseasePrediction(BaseModel):
    disease: str
    probability: str
    description: str
    precautions: List[str]

class PredictionResponse(BaseModel):
    predictions: List[DiseasePrediction]
    disclaimer: str

@app.get("/")
async def root():
    return {
        "message": "AI Disease Prediction API",
        "version": "1.0.0",
        "description": "Predict diseases based on symptoms using machine learning"
    }

@app.get("/symptoms")
async def get_symptoms():
    """Get list of all available symptoms"""
    if not symptoms_list:
        raise HTTPException(status_code=500, detail="Symptoms data not loaded")
    return {"symptoms": symptoms_list}

@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(request: SymptomRequest):
    """Predict diseases based on symptoms"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    if not symptoms_list:
        raise HTTPException(status_code=500, detail="Symptoms data not loaded")
    
    # Validate symptoms
    invalid_symptoms = [s for s in request.symptoms if s not in symptoms_list]
    if invalid_symptoms:
        raise HTTPException(status_code=400, detail=f"Invalid symptoms: {invalid_symptoms}")
    
    if len(request.symptoms) == 0:
        raise HTTPException(status_code=400, detail="At least one symptom must be provided")
    
    # Create feature vector
    feature_vector = np.zeros(len(symptoms_list))
    for symptom in request.symptoms:
        if symptom in symptoms_list:
            idx = symptoms_list.index(symptom)
            # Apply severity weighting (scale 1-5)
            feature_vector[idx] = request.severity / 5.0
    
    # Make prediction
    try:
        # Get probability scores for all classes
        probabilities = model.predict_proba([feature_vector])[0]
        classes = model.classes_
        
        # Create list of (disease, probability) pairs
        disease_probs = list(zip(classes, probabilities))
        
        # Sort by probability (descending) and get top 3
        disease_probs.sort(key=lambda x: x[1], reverse=True)
        top_3 = disease_probs[:3]
        
        # Format predictions
        predictions = []
        for disease, prob in top_3:
            info = disease_info.get(disease, {
                "description": "No description available",
                "precautions": ["Consult a healthcare professional"]
            })
            
            predictions.append(DiseasePrediction(
                disease=disease,
                probability=f"{prob*100:.1f}%",
                description=info["description"],
                precautions=info["precautions"]
            ))
        
        disclaimer = "⚠️ This tool is for educational purposes only and not a substitute for professional medical advice. Always consult a healthcare professional for proper diagnosis and treatment."
        
        return PredictionResponse(
            predictions=predictions,
            disclaimer=disclaimer
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "symptoms_available": len(symptoms_list) if symptoms_list else 0,
        "label_encoder_loaded": label_encoder is not None,
        "timestamp": datetime.now().isoformat()
    }

# CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)