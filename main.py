import numpy as np
import pandas as pd
import joblib
import argparse
import logging
import os
import mlflow
import mlflow.sklearn
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Définir l'URI de suivi MLflow (local)
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Churn Prediction")

def main():
    """Gestion des arguments en ligne de commande."""
    parser = argparse.ArgumentParser(description="Pipeline de Machine Learning pour la prédiction du churn avec Random Forest")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle avec suivi MLflow")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle et enregistrer les métriques")
    parser.add_argument("--data_path", type=str, default="merged_churn.csv", help="Chemin du fichier de données")
    args = parser.parse_args()

    if args.prepare:
        logging.info("🔄 Préparation des données...")
        X_train, X_test, y_train, y_test = prepare_data(args.data_path)
        logging.info(f"✅ Données préparées : {X_train.shape[0]} échantillons d'entraînement, {X_test.shape[0]} de test.")

    if args.train:
        logging.info("🚀 Entraînement du modèle avec suivi MLflow...")
        X_train, X_test, y_train, y_test= prepare_data(args.data_path)

        with mlflow.start_run():
            model = train_model(X_train, y_train)

            # Log des métriques sur le test set
            test_accuracy = evaluate_model(model, X_test, y_test)
            mlflow.log_metric("test_accuracy", test_accuracy)

            save_model(model)
        logging.info("✅ Modèle entraîné et enregistré.")

    if args.evaluate:
        logging.info("📊 Évaluation du modèle existant...")
        X_train, X_test, y_train, y_test = prepare_data(args.data_path)
        model = load_model()
        
        if model:
            test_accuracy = evaluate_model(model, X_test, y_test)
            logging.info(f"✅ Précision du modèle sur le test set : {test_accuracy:.4f}")
        else:
            logging.error("❌ Impossible de charger le modèle pour l'évaluation.")

if __name__ == "__main__":
    main()
