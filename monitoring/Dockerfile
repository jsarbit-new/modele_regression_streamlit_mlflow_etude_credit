# Dockerfile dans C:\Users\jonjo\Documents\open classrooms\Projet 7\monitoring
FROM python:3.9-slim-buster

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Copie le fichier requirements.txt dans le répertoire de travail du conteneur
COPY requirements.txt .

# Installe toutes les dépendances listées dans requirements.txt
# --no-cache-dir réduit la taille de l'image finale
RUN pip install --no-cache-dir -r requirements.txt

# Copie votre script Python dans le répertoire de travail du conteneur
COPY generate_report.py .

# Commande par défaut à exécuter lorsque le conteneur démarre
CMD ["python", "generate_report.py"]