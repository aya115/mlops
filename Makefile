VENV_NAME = venv

# Installer les dépendances
install:
	@echo "Installation des dépendances..."
	@pip install -r requirements.txt

# Créer un environnement virtuel
venv:
	@echo "Création de l'environnement virtuel..."
	@python3 -m venv $(VENV_NAME)

# Activer l'environnement virtuel
activate:
	@echo "Activez votre environnement virtuel avec la commande suivante :"
	@echo "source $(VENV_NAME)/bin/activate"

# Vérification du code
lint:
	@echo "Vérification du code avec flake8..."
	@flake8 --max-line-length=120

# Préparer les données
prepare:
	@echo "Préparation des données..."
	@python3 main.py --prepare

# Entraîner le modèle avec MLflow
train:
	@echo "Entraînement du modèle avec MLflow..."
	@python3 main.py --train

# Évaluer le modèle
evaluate:
	@echo "Évaluation du modèle..."
	@python3 main.py --evaluate

# Lancer l'interface MLflow
mlflow-ui:
	@echo "Démarrage de l'interface MLflow..."
	@mlflow ui --host 0.0.0.0 --port 5000 &

# Exécuter les tests
test:
	@echo "Exécution des tests..."
	@python -m pytest --maxfail=3 --disable-warnings -q

# Nettoyage des fichiers inutiles
clean:
	@echo "Nettoyage des fichiers inutiles..."
	@rm -rf pycache *.pyc $(VENV_NAME)

# Commande par défaut qui prépare l'environnement et entraîne le modèle
all:
	install train

test_api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000 &
	sleep 3 && python test_api.py


IMAGE_NAME=soltani-aya-4ds5-mlflow-app
build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -p 8000:8000 $(IMAGE_NAME)

tag:
	docker tag $(IMAGE_NAME) moncompte_dockerhub/$(IMAGE_NAME)

push:
	docker push moncompte_dockerhub/$(IMAGE_NAME)

clean_docker:
	docker rmi $(IMAGE_NAME)
