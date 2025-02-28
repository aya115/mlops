import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. Configurer MLflow pour utiliser un stockage local
mlflow.set_tracking_uri("file:///home/aya/soltani-aya-4ds5-ml_project/mlruns")  # Utiliser le répertoire local
mlflow.set_experiment("Mon_Experience")  # Nom de l'expérience

# 2. Charger les données
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# 3. Entraîner un modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 4. Enregistrer le modèle avec MLflow
with mlflow.start_run():
    # Enregistrer le modèle
    mlflow.sklearn.log_model(model, "model")

    # Enregistrer une métrique (ex: précision)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Enregistrer des paramètres (ex: hyperparamètres du modèle)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)

    # Enregistrer un artefact (ex: un fichier texte)
    with open("info.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}")
    mlflow.log_artifact("info.txt")

print(f"Run enregistré avec succès. Précision du modèle : {accuracy}")
