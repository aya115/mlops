import os
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Définition des fichiers pour sauvegarder le modèle et le scaler
MODEL_FILENAME = "models/model.pkl"
SCALER_FILENAME = "models/scaler.pkl"
FEATURES_FILENAME = "models/model_features.pkl"

# Sélection des 16 features à utiliser
SELECTED_FEATURES = [
    "Account length", "Area code", "Customer service calls", "International plan",
    "Number vmail messages", "Total day calls", "Total day charge", "Total day minutes",
    "Total night calls", "Total night charge", "Total night minutes",
    "Total eve calls", "Total eve charge", "Total eve minutes",
    "Total intl calls", "Voice mail plan"
]

def prepare_data(data_path='merged_churn.csv'):
    """Chargement, prétraitement et séparation des données."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Erreur : le fichier {data_path} est introuvable.")

    data = pd.read_csv(data_path)
    
    # Encodage des variables catégoriques
    label_encoder = LabelEncoder()
    for col in ["International plan", "Voice mail plan", "Churn"]:
        data[col] = label_encoder.fit_transform(data[col])
    
    X = data[SELECTED_FEATURES]
    y = data['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Création du dossier 'models' s'il n'existe pas
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, SCALER_FILENAME)
    joblib.dump(SELECTED_FEATURES, FEATURES_FILENAME)
    
    print(f"✅ Données préparées : {X_train.shape[0]} échantillons d'entraînement, {X_test.shape[0]} de test.")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Entraîne un modèle Gradient Boosting et l'enregistre avec MLflow."""
    
    # 🚨 Ferme tout run MLflow actif pour éviter l'erreur
    mlflow.end_run()

    with mlflow.start_run():
        gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, subsample=1.0, random_state=42)
        gb_model.fit(X_train, y_train)

        # Enregistrement des hyperparamètres
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("subsample", 1.0)

        # Enregistrement du modèle
        save_model(gb_model)
        mlflow.sklearn.log_model(gb_model, "model")
        mlflow.log_artifact(MODEL_FILENAME)

        print("✅ Modèle entraîné et enregistré avec MLflow.")
        return gb_model

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle et enregistre la métrique d'accuracy dans MLflow."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n📊 Classification Report :")
    print(classification_report(y_test, y_pred))
    print(f"✅ Précision du modèle : {accuracy:.4f}")
    mlflow.end_run()  # Ferme tout run actif avant d'en démarrer un nouveau

    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_artifact(FEATURES_FILENAME)

    return accuracy

def save_model(model, filename=MODEL_FILENAME):
    """Enregistre le modèle dans un fichier pickle."""
    joblib.dump(model, filename)
    print(f"✅ Modèle enregistré sous {filename}")

def load_model(filename=MODEL_FILENAME):
    """Charge un modèle enregistré."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ Le fichier {filename} est introuvable.")
    print(f"✅ Modèle chargé depuis {filename}")
    return joblib.load(filename)

