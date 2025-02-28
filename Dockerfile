# Utilisation d'une image Python légère
FROM python:3.12

# Définition du répertoire de travail
WORKDIR /app

# Copier tout le projet dans l'image Docker
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par FastAPI
EXPOSE 8000

# Démarrer l'API FastAPI depuis app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
