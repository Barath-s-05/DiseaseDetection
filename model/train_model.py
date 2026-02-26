import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create comprehensive symptom dataset
def create_symptom_dataset():
    # Common symptoms
    symptoms = [
        'fever', 'cough', 'headache', 'fatigue', 'nausea', 'vomiting',
        'diarrhea', 'abdominal_pain', 'chest_pain', 'shortness_of_breath',
        'dizziness', 'joint_pain', 'muscle_pain', 'sore_throat', 'runny_nose',
        'skin_rash', 'itching', 'swelling', 'weight_loss', 'weight_gain',
        'loss_of_appetite', 'excessive_thirst', 'frequent_urination',
        'blurred_vision', 'tingling', 'numbness', 'confusion',
        'memory_problems', 'mood_changes', 'sleep_disturbance'
    ]
    
    # Diseases with their typical symptoms
    disease_data = [
        # Common Cold
        {'disease': 'Common Cold', 'symptoms': ['cough', 'sore_throat', 'runny_nose', 'headache', 'fatigue'], 'description': 'A viral infection of the upper respiratory tract.', 'precautions': ['Rest adequately', 'Drink plenty of fluids', 'Use saline nasal drops', 'Avoid close contact with others']},
        
        # Influenza (Flu)
        {'disease': 'Influenza', 'symptoms': ['fever', 'cough', 'fatigue', 'headache', 'muscle_pain'], 'description': 'A contagious respiratory illness caused by influenza viruses.', 'precautions': ['Get annual flu vaccine', 'Rest and stay hydrated', 'Take antiviral medications if prescribed', 'Isolate to prevent spreading']},
        
        # COVID-19
        {'disease': 'COVID-19', 'symptoms': ['fever', 'cough', 'shortness_of_breath', 'fatigue', 'loss_of_appetite'], 'description': 'A respiratory disease caused by the SARS-CoV-2 virus.', 'precautions': ['Get vaccinated', 'Wear mask in public', 'Maintain social distancing', 'Isolate if symptomatic']},
        
        # Migraine
        {'disease': 'Migraine', 'symptoms': ['headache', 'nausea', 'dizziness', 'sensitivity_to_light', 'confusion'], 'description': 'A neurological condition characterized by intense headache.', 'precautions': ['Identify and avoid triggers', 'Maintain regular sleep schedule', 'Stay hydrated', 'Practice stress management']},
        
        # Hypertension
        {'disease': 'Hypertension', 'symptoms': ['headache', 'dizziness', 'chest_pain', 'shortness_of_breath'], 'description': 'High blood pressure that can lead to serious health problems.', 'precautions': ['Monitor blood pressure regularly', 'Reduce sodium intake', 'Exercise regularly', 'Take prescribed medications']},
        
        # Diabetes
        {'disease': 'Diabetes', 'symptoms': ['excessive_thirst', 'frequent_urination', 'fatigue', 'blurred_vision', 'weight_loss'], 'description': 'A group of metabolic disorders characterized by high blood sugar.', 'precautions': ['Monitor blood glucose levels', 'Follow diabetic diet', 'Take prescribed medications', 'Regular exercise']},
        
        # Asthma
        {'disease': 'Asthma', 'symptoms': ['shortness_of_breath', 'cough', 'chest_tightness', 'wheezing'], 'description': 'A chronic inflammatory disease of the airways.', 'precautions': ['Avoid triggers', 'Use inhaler as prescribed', 'Monitor breathing', 'Emergency action plan']},
        
        # Gastroenteritis
        {'disease': 'Gastroenteritis', 'symptoms': ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain', 'fever'], 'description': 'Inflammation of the stomach and intestines.', 'precautions': ['Stay hydrated', 'BRAT diet (bananas, rice, applesauce, toast)', 'Rest', 'Avoid dairy and fatty foods']},
        
        # Arthritis
        {'disease': 'Arthritis', 'symptoms': ['joint_pain', 'swelling', 'stiffness', 'fatigue'], 'description': 'Inflammation of one or more joints.', 'precautions': ['Regular low-impact exercise', 'Maintain healthy weight', 'Hot/cold therapy', 'Anti-inflammatory medications']},
        
        # Depression
        {'disease': 'Depression', 'symptoms': ['fatigue', 'sleep_disturbance', 'loss_of_appetite', 'mood_changes', 'confusion'], 'description': 'A mood disorder causing persistent feelings of sadness.', 'precautions': ['Seek professional help', 'Regular exercise', 'Maintain social connections', 'Follow treatment plan']},
        
        # Anxiety Disorder
        {'disease': 'Anxiety Disorder', 'symptoms': ['dizziness', 'shortness_of_breath', 'chest_pain', 'confusion', 'sleep_disturbance'], 'description': 'Excessive worry and fear that interferes with daily activities.', 'precautions': ['Practice relaxation techniques', 'Regular exercise', 'Limit caffeine', 'Professional counseling']},
        
        # Allergic Reaction
        {'disease': 'Allergic Reaction', 'symptoms': ['skin_rash', 'itching', 'swelling', 'difficulty_breathing'], 'description': 'Immune system response to allergens.', 'precautions': ['Identify and avoid allergens', 'Take antihistamines', 'Carry epinephrine auto-injector', 'Emergency medical attention']},
        
        # Pneumonia
        {'disease': 'Pneumonia', 'symptoms': ['fever', 'cough', 'shortness_of_breath', 'chest_pain', 'fatigue'], 'description': 'Infection that inflames air sacs in one or both lungs.', 'precautions': ['Complete antibiotic course', 'Rest and hydration', 'Vaccination (pneumococcal)', 'Follow-up care']},
        
        # Bronchitis
        {'disease': 'Bronchitis', 'symptoms': ['cough', 'chest_discomfort', 'fatigue', 'shortness_of_breath'], 'description': 'Inflammation of the bronchial tubes.', 'precautions': ['Stay hydrated', 'Use humidifier', 'Avoid smoke', 'Rest']},
        
        # Sinusitis
        {'disease': 'Sinusitis', 'symptoms': ['headache', 'facial_pain', 'nasal_congestion', 'fever'], 'description': 'Inflammation of the tissue lining the sinuses.', 'precautions': ['Saline nasal irrigation', 'Steam inhalation', 'Decongestants', 'Antibiotics if bacterial']},
        
        # Urinary Tract Infection
        {'disease': 'UTI', 'symptoms': ['frequent_urination', 'burning_urination', 'abdominal_pain', 'fever'], 'description': 'Infection in any part of the urinary system.', 'precautions': ['Drink plenty of water', 'Cranberry juice', 'Complete antibiotic course', 'Good hygiene']},
        
        # Gastroesophageal Reflux
        {'disease': 'GERD', 'symptoms': ['heartburn', 'chest_pain', 'nausea', 'difficulty_swallowing'], 'description': 'Chronic digestive disease with stomach acid flowing back.', 'precautions': ['Avoid trigger foods', 'Elevate head while sleeping', 'Small frequent meals', 'Antacids or prescribed medication']},
        
        # Anemia
        {'disease': 'Anemia', 'symptoms': ['fatigue', 'dizziness', 'shortness_of_breath', 'pale_skin'], 'description': 'Condition with insufficient healthy red blood cells.', 'precautions': ['Iron-rich diet', 'Vitamin supplements', 'Treat underlying cause', 'Regular monitoring']},
        
        # Thyroid Disorder
        {'disease': 'Thyroid Disorder', 'symptoms': ['weight_changes', 'fatigue', 'mood_changes', 'temperature_sensitivity'], 'description': 'Disorder of the thyroid gland affecting metabolism.', 'precautions': ['Regular thyroid function tests', 'Medication compliance', 'Balanced diet', 'Stress management']},
        
        # Food Poisoning
        {'disease': 'Food Poisoning', 'symptoms': ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain', 'fever'], 'description': 'Illness caused by eating contaminated food.', 'precautions': ['Stay hydrated', 'BRAT diet', 'Rest', 'Seek medical attention if severe']}
    ]
    
    # Generate dataset
    data = []
    for disease_info in disease_data:
        # Create multiple samples per disease with some variation
        for _ in range(15):  # 15 samples per disease
            row = {}
            
            # Set all symptoms to 0 initially
            for symptom in symptoms:
                row[symptom] = 0
            
            # Set actual symptoms to 1 (with some random variation)
            actual_symptoms = disease_info['symptoms']
            for symptom in actual_symptoms:
                if symptom in symptoms:
                    # 80-100% chance of having the symptom
                    row[symptom] = 1 if np.random.random() > 0.2 else 0
            
            # Add some random symptoms (10% chance)
            for symptom in symptoms:
                if symptom not in actual_symptoms and np.random.random() < 0.1:
                    row[symptom] = 1
            
            row['disease'] = disease_info['disease']
            data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('symptom_data.csv', index=False)
    
    return df, symptoms, disease_data

# Train models
def train_models(df, symptoms):
    # Prepare features and target
    X = df[symptoms]
    y = df['disease']
    
    # Encode labels for models that require numerical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Also split encoded labels
    _, _, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    print("Training models...")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of diseases: {len(y.unique())}")
    
    models = {}
    scores = {}
    
    # Random Forest
    print("\nTraining Random Forest...")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)
    models['random_forest'] = rf.best_estimator_
    scores['random_forest'] = rf_score
    print(f"Random Forest Accuracy: {rf_score:.4f}")
    
    # Gradient Boosting
    print("\nTraining Gradient Boosting...")
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.05],
        'max_depth': [3, 5]
    }
    gb = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=5, scoring='accuracy')
    gb.fit(X_train, y_train)
    gb_score = gb.score(X_test, y_test)
    models['gradient_boosting'] = gb.best_estimator_
    scores['gradient_boosting'] = gb_score
    print(f"Gradient Boosting Accuracy: {gb_score:.4f}")
    
    # Try XGBoost
    try:
        from xgboost import XGBClassifier
        print("\nTraining XGBoost...")
        xgb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.05],
            'max_depth': [3, 6]
        }
        xgb = GridSearchCV(XGBClassifier(random_state=42), xgb_params, cv=5, scoring='accuracy')
        # Use encoded labels for XGBoost
        xgb.fit(X_train, y_train_encoded)
        # Calculate score using encoded test labels
        xgb_pred_encoded = xgb.predict(X_test)
        xgb_score = accuracy_score(y_test_encoded, xgb_pred_encoded)
        # Convert the XGBoost model to a wrapper that handles label encoding internally
        class XGBoostWrapper:
            def __init__(self, model, label_encoder):
                self.model = model
                self.label_encoder = label_encoder
            
            def predict(self, X):
                pred_encoded = self.model.predict(X)
                return self.label_encoder.inverse_transform(pred_encoded)
            
            def predict_proba(self, X):
                return self.model.predict_proba(X)
            
            @property
            def classes_(self):
                return self.label_encoder.classes_
        
        xgb_wrapped = XGBoostWrapper(xgb.best_estimator_, label_encoder)
        models['xgboost'] = xgb_wrapped
        scores['xgboost'] = xgb_score
        print(f"XGBoost Accuracy: {xgb_score:.4f}")
    except ImportError:
        print("XGBoost not available, skipping...")
    
    # Select best model
    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name} with accuracy: {scores[best_model_name]:.4f}")
    
    # Save model and data
    joblib.dump(best_model, 'disease_model.pkl')
    joblib.dump(symptoms, 'symptoms_list.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')  # Save the label encoder
    
    return best_model, symptoms, scores

if __name__ == "__main__":
    # Create dataset
    df, symptoms, disease_info = create_symptom_dataset()
    print(f"Dataset created with {len(df)} samples and {len(symptoms)} symptoms")
    print(f"Diseases: {df['disease'].nunique()}")
    
    # Train models
    best_model, symptom_list, scores = train_models(df, symptoms)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Model saved as: disease_model.pkl")
    print(f"Symptoms list saved as: symptoms_list.pkl")
    print(f"Dataset saved as: symptom_data.csv")