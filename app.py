from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Chargement du modèle, du scaler et des features
MODEL_FILENAME = "decision_tree_model.pkl"
SCALER_FILENAME = "scaler.pkl"
FEATURES_FILENAME = "model_features.pkl"

try:
    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    SELECTED_FEATURES = joblib.load(FEATURES_FILENAME)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement des fichiers : {e}")

# Initialisation de l'API FastAPI
app = FastAPI()

# Modèle de requête pour la prédiction
class PredictionInput(BaseModel):
    features: list  # Liste des features en entrée

# Encodage des variables catégorielles
label_encoder = LabelEncoder()

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # Vérification du nombre de features
        if len(data.features) != len(SELECTED_FEATURES):
            raise HTTPException(
                status_code=400, 
                detail=f"Nombre incorrect de features. Attendu : {len(SELECTED_FEATURES)}, Reçu : {len(data.features)}"
            )

        # Création d'un DataFrame avec les features
        input_data = pd.DataFrame([data.features], columns=SELECTED_FEATURES)

        # Vérification si les colonnes catégoriques existent avant encodage
        for col in ["International plan", "Voice mail plan"]:
            if col in input_data.columns:
                input_data[col] = label_encoder.fit_transform(input_data[col])

        # Normalisation des features
        scaled_features = scaler.transform(input_data)

        # Prédiction
        prediction = model.predict(scaled_features)
        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {e}")

