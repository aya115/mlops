from flask import Flask, render_template, request
import requests
import joblib
import os

app = Flask(__name__)

API_URL = "http://127.0.0.1:8000/predict"

# Chargement du scaler et des features
scaler_path = 'models/scaler.pkl'
model_features_path = 'models/model_features.pkl'

if os.path.exists(scaler_path) and os.path.exists(model_features_path):
    scaler = joblib.load(scaler_path)  # Chargement du scaler
    selected_features = joblib.load(model_features_path)  # Chargement des features utilisées par le modèle
else:
    raise FileNotFoundError("Les fichiers du modèle ou du scaler sont introuvables.")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Conversion et récupération des valeurs du formulaire
            international_plan = 1 if request.form["international_plan"] == "yes" else 0
            voice_mail_plan = 1 if request.form["voice_mail_plan"] == "yes" else 0

            input_data = [
                int(request.form["account_length"]),
                int(request.form["area_code"]),
                int(request.form["customer_service_calls"]),
                international_plan,
                int(request.form["number_vmail_messages"]),
                int(request.form["total_day_calls"]),
                float(request.form["total_day_charge"]),
                float(request.form["total_day_minutes"]),
                int(request.form["total_night_calls"]),
                float(request.form["total_night_charge"]),
                float(request.form["total_night_minutes"]),
                int(request.form["total_eve_calls"]),
                float(request.form["total_eve_charge"]),
                float(request.form["total_eve_minutes"]),
                int(request.form["total_intl_calls"]),
                voice_mail_plan
            ]

            # Vérification du nombre de features
            if len(input_data) != len(selected_features):
                return f"Erreur : Nombre de features incorrect ({len(input_data)} reçues, {len(selected_features)} attendues)."

            # Standardisation des données
            input_data_scaled = scaler.transform([input_data])

            # Envoi à l'API
            response = requests.post(API_URL, json={"features": input_data_scaled.tolist()[0]})

            if response.status_code == 200:
                prediction = response.json().get("prediction", "Aucune prédiction reçue.")
            else:
                prediction = f"Erreur API : {response.text}"

        except ValueError as e:
            prediction = f"Erreur de conversion des données : {str(e)}"
        except KeyError as e:
            prediction = f"Erreur : Champ manquant {str(e)}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

