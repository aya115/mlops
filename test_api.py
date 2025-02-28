import requests

# URL de l'API FastAPI
API_URL = "http://127.0.0.1:8000/predict"

# Liste des valeurs dans le bon ordre
input_data = {
    "features": [
        107,   # Account length
        415,   # Area code
        1,     # Customer service calls
        "no",  # International plan (should be encoded)
        26,    # Number vmail messages
        123,   # Total day calls
        27.47, # Total day charge
        161.6, # Total day minutes
        103,   # Total night calls
        11.45, # Total night charge
        254.4, # Total night minutes
        103,   # Total eve calls
        16.62, # Total eve charge
        195.5, # Total eve minutes
        3,     # Total intl calls
        "yes"  # Voice mail plan (should be encoded)
    ]
}

# Envoyer une requête POST à l'API
response = requests.post(API_URL, json=input_data)

# Vérifier le résultat
if response.status_code == 200:
    print("✅ Test réussi ! Réponse :", response.json())
else:
    print("❌ Test échoué ! Code :", response.status_code, "Détails :", response.text)

