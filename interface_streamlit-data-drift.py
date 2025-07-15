import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import json
import logging
import os
import streamlit.components.v1 as components # Importation pour st.components.v1.html

# # Imports Evidently - Ces lignes ne sont plus nécessaires directement dans app.py
# from evidently.report import Report
# from evidently.metric_preset import DataDriftPreset

# --- 1. Configuration de la Page Streamlit (DOIT ABSOLUMENT ÊTRE LA PREMIÈRE COMMANDE ST.) ---
st.set_page_config(
    page_title="Prédiction de Défaut Client & Surveillance",
    page_icon="📊",
    layout="wide"
)

# --- 2. Configuration du Logging pour Streamlit ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 3. Configuration MLflow pour l'Application ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_MODEL_NAME = "HomeCreditLogisticRegressionPipeline"

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"Streamlit: MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
except Exception as e:
    st.error(f"Streamlit: Erreur critique lors de la configuration de MLflow Tracking URI: {e}")
    st.info("Vérifiez que votre serveur MLflow est démarré et accessible à l'adresse spécifiée.")
    st.stop()

# --- 4. Fonction de Chargement du Modèle et des Métadonnées (Mise en Cache) ---
@st.cache_resource(show_spinner="Chargement du modèle de prédiction et de ses métadonnées depuis MLflow...")
def load_mlflow_model_and_metadata():
    """
    Charge le modèle et ses métadonnées (features_info, seuil optimal, all_training_features)
    depuis MLflow Model Registry. Cette fonction est mise en cache pour n'être exécutée qu'une seule fois.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        model_versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")

        if not model_versions:
            error_msg = (f"Aucune version du modèle '{MLFLOW_MODEL_NAME}' trouvée dans le Model Registry MLflow. "
                         "Assurez-vous que le script d'entraînement a été exécuté et que le modèle "
                         "a été enregistré avec le nom spécifié.")
            logger.error(f"Streamlit: {error_msg}")
            raise ValueError(error_msg)

        latest_version = max(model_versions, key=lambda mv: mv.version)
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/{latest_version.version}"

        model = mlflow.sklearn.load_model(model_uri=model_uri)
        logger.info(f"Streamlit: Modèle '{MLFLOW_MODEL_NAME}' (Version {latest_version.version}) chargé avec succès en tant que Scikit-learn model.")

        features_info = None
        optimal_threshold = None
        all_training_features = None

        # Tenter de récupérer des tags (préférable)
        model_metadata_tags = latest_version.tags if hasattr(latest_version, 'tags') else {}
        optimal_threshold_str = model_metadata_tags.get('mlflow.log_model.metadata.optimal_threshold')
        features_info_str = model_metadata_tags.get('mlflow.log_model.metadata.features_info_for_streamlit')
        all_training_features_str = model_metadata_tags.get('mlflow.log_model.metadata.all_training_features')

        if optimal_threshold_str:
            optimal_threshold = float(optimal_threshold_str)
            logger.info(f"Streamlit: Seuil optimal récupéré des tags du modèle: {optimal_threshold:.4f}")
        else:
            logger.warning("Streamlit: Seuil optimal non trouvé dans les tags du modèle. Tentative de fallback...")

        if features_info_str:
            features_info = json.loads(features_info_str)
            logger.info(f"Streamlit: Informations sur les features récupérées des tags du modèle ({len(features_info)} features).")
        else:
            logger.warning("Streamlit: Informations sur les features non trouvées dans les tags du modèle. Tentative de fallback...")

        if all_training_features_str:
            all_training_features = json.loads(all_training_features_str)
            logger.info(f"Streamlit: Liste complète des features d'entraînement récupérée des tags du modèle ({len(all_training_features)} features).")
        else:
            logger.warning("Streamlit: Liste complète des features d'entraînement non trouvée dans les tags du modèle. Tentative de fallback...")

        # Fallback aux paramètres de la run si les tags ne sont pas présents
        if features_info is None or optimal_threshold is None or all_training_features is None:
            logger.warning("Streamlit: Tentative de récupération des métadonnées manquantes depuis les paramètres de la run MLflow...")
            run_info = client.get_run(latest_version.run_id)
            run_info_params = run_info.data.params

            if optimal_threshold is None and 'optimal_threshold_value' in run_info_params:
                optimal_threshold = float(run_info_params['optimal_threshold_value'])
                logger.info(f"Streamlit: Seuil optimal récupéré des paramètres de la run: {optimal_threshold:.4f}")

            if features_info is None and 'features_info_for_streamlit_json' in run_info_params:
                features_info_json = run_info_params['features_info_for_streamlit_json']
                features_info = json.loads(features_info_json)
                logger.info(f"Streamlit: Informations sur les features récupérées des paramètres de la run ({len(features_info)} features).")

            if all_training_features is None and 'all_training_features_names' in run_info_params:
                all_training_features_json = run_info_params['all_training_features_names']
                all_training_features = json.loads(all_training_features_json)
                logger.info(f"Streamlit: Liste complète des features d'entraînement récupérée des paramètres de la run ({len(all_training_features)} features).")

        if features_info is None or not features_info:
            raise ValueError("Impossible de déterminer la liste des variables d'entrée importantes pour l'interface. "
                             "Veuillez vous assurer qu'elles sont logguées correctement lors de l'entraînement.")

        if optimal_threshold is None:
            optimal_threshold = 0.5
            logger.warning(f"Streamlit: Seuil optimal non trouvé du tout. Utilisation d'un seuil par défaut de {optimal_threshold}.")

        if all_training_features is None or not all_training_features:
            logger.warning("Liste complète des features d'entraînement ('all_training_features') non trouvée. "
                            "Cela pourrait causer des problèmes si les features d'entrée ne sont pas ordonnées comme prévu.")
            all_training_features = list(features_info.keys()) # Fallback: utiliser les features importantes

        return model, features_info, optimal_threshold, all_training_features

    except Exception as e:
        st.error(f"Streamlit: Échec critique lors du chargement du modèle ou de ses métadonnées: {e}")
        st.info("""
            **Vérifiez les points suivants pour résoudre le problème :**
            1.  **Serveur MLflow UI démarré :** Assurez-vous que `mlflow ui` (avec le bon `--backend-store-uri` si nécessaire) est bien en cours d'exécution dans un terminal séparé.
            2.  **Modèle enregistré :** Vérifiez que le modèle nommé `HomeCreditLogisticRegressionPipeline` est visible dans l'onglet "Models" de l'interface MLflow UI et qu'il a des **tags/métadonnées** pour 'optimal_threshold' et 'features_info_for_streamlit'.
            3.  **Logs d'entraînement :** Examinez les logs de votre script d'entraînement pour confirmer que le modèle a été enregistré sans erreur et que la liste des features et le seuil optimal ont été loggués correctement.
        """)
        st.stop()
        return None, None, None, None

# --- 5. Chargement des Ressources au Démarrage de l'Application Streamlit ---
model, features_info, optimal_threshold, all_training_features = load_mlflow_model_and_metadata()


# --- Nouvelle fonction pour afficher le rapport Evidently (lecture seule) ---
# Pas besoin de st.cache_data ici car le fichier est censé être généré à l'extérieur
def display_evidently_report(report_file_path):
    """
    Charge et affiche le rapport Evidently HTML depuis un fichier.
    """
    if not os.path.exists(report_file_path):
        st.warning(f"Le rapport Evidently n'a pas encore été généré. Veuillez d'abord l'exécuter via Docker.")
        st.info(f"Pour générer le rapport, ouvrez un terminal dans le dossier `{os.path.dirname(report_file_path)}` "
                f"et exécutez les commandes Docker comme expliqué ci-dessous.")
        return

    try:
        with open(report_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=1000, scrolling=True)
        st.success(f"Rapport Evidently chargé et affiché avec succès depuis : `{report_file_path}`")
    except Exception as e:
        st.error(f"Erreur lors de l'affichage du rapport Evidently : {e}")
        logger.exception("Erreur lors de l'affichage du rapport Evidently dans Streamlit:")


# --- Chemin où le rapport Evidently sera sauvegardé et lu ---
# Important : ce chemin doit être accessible par Streamlit localement
# et doit être le même que le chemin de sortie du Docker
EVLIDENTLY_REPORT_PATH = "evidently_data_drift_report.html" # A la racine de votre projet

# --- 6. Contenu Principal de la Page Streamlit ---
st.title("📊 Prédiction de Défaut Client & Surveillance du Modèle")

# Création des onglets
tab1, tab2 = st.tabs(["Prédiction de Prêt", "Analyse du Data Drift"])

with tab1:
    st.markdown("""
    Cette application interactive vous permet de simuler une prédiction de risque de défaut pour un client.
    Entrez les valeurs des caractéristiques ci-dessous et le modèle calculera la probabilité de défaut,
    puis classifiera le client comme à "Risque Élevé" ou "Risque Faible" en utilisant le seuil optimal déterminé lors de l'entraînement.
    """)

    # Affichage des informations sur le modèle chargé dans une barre latérale pour le débogage et l'information
    if model and features_info:
        st.sidebar.header("Informations sur le Modèle")
        st.sidebar.write(f"**Nom du Modèle :** `{MLFLOW_MODEL_NAME}`")
        try:
            # Tente de récupérer la version du modèle pour l'affichage
            client = mlflow.tracking.MlflowClient()
            current_version_obj = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=['None', 'Production', 'Staging'])
            current_version = current_version_obj[0].version if current_version_obj else "N/A"
            st.sidebar.write(f"**Version du Modèle :** `{current_version}`")
        except Exception as e:
            st.sidebar.warning(f"Impossible de récupérer la version du modèle: {e}")
        st.sidebar.write(f"**Seuil Optimal Utilisé :** `{optimal_threshold:.4f}`")
        st.sidebar.write(f"**Nombre de Caractéristiques Importantes :** `{len(features_info)}`")
        st.sidebar.info("Vérifiez les logs de votre terminal pour plus de détails sur le chargement.")

    # --- 7. Interface de Saisie des Caractéristiques ---
    if model is not None and features_info is not None and all_training_features is not None:
        st.subheader("Saisie des Caractéristiques Client Importantes")
        st.markdown("Veuillez remplir les champs ci-dessous. Les valeurs par défaut sont celles du dataset (ou 0.0 si non spécifié). "
                    "Vous pouvez les modifier pour voir l'impact sur la prédiction. "
                    "Les plages de valeurs sont indicatives, basées sur l'entraînement.")

        user_inputs = {}
        num_columns = 2
        cols = st.columns(num_columns)
        col_idx = 0

        for feature_original_name, info in features_info.items():
            display_name = info.get("display_name", feature_original_name)
            min_val_hint = info.get("min_val")
            max_val_hint = info.get("max_val")

            default_value = (float(min_val_hint) + float(max_val_hint)) / 2 if min_val_hint is not None and max_val_hint is not None else 0.0

            with cols[col_idx]:
                st.write(f"**{display_name}**")
                st.caption(f"Plage: [{min_val_hint:.2f} - {max_val_hint:.2f}]" if min_val_hint is not None and max_val_hint is not None else "Plage: N/A")

                user_inputs[feature_original_name] = st.number_input(
                    label=" ", # Le label est mis à vide car il est déjà affiché au-dessus
                    value=float(default_value),
                    min_value=float(min_val_hint) if min_val_hint is not None else None,
                    max_value=float(max_val_hint) if max_val_hint is not None else None,
                    format="%.4f",
                    key=f"input_{feature_original_name}" # Clé unique pour chaque widget
                )
            col_idx = (col_idx + 1) % num_columns

        st.markdown("---")

        # --- 8. Bouton de Prédiction et Affichage des Résultats ---
        if st.button("Obtenir la Prédiction", help="Cliquez pour exécuter le modèle avec les valeurs saisies."):
            model_input_data = {}
            for feature_name in all_training_features:
                if feature_name in user_inputs:
                    model_input_data[feature_name] = user_inputs[feature_name]
                else:
                    # Gérer les features non-importantes ou manquantes en leur donnant une valeur par défaut
                    model_input_data[feature_name] = 0.0

            input_df = pd.DataFrame([model_input_data])
            input_df = input_df[all_training_features]

            try:
                prediction_proba = model.predict_proba(input_df)[:, 1][0]
                prediction_class = 1 if prediction_proba >= optimal_threshold else 0

                st.subheader("🎉 Résultat de la Prédiction :")

                col_proba, col_class = st.columns(2)
                with col_proba:
                    st.metric(label="Probabilité de Défaut", value=f"{prediction_proba:.4f}")

                with col_class:
                    if prediction_class == 1:
                        st.error("⚠️ **Client à Risque de Défaut Élevé**")
                    else:
                        st.success("✅ **Client à Risque de Défaut Faible**")

                st.markdown("---")
                st.write(f"**Détails de l'analyse :**")
                st.markdown(f"- Le modèle a calculé une probabilité de défaut de **`{prediction_proba:.4f}`**.")
                st.markdown(f"- Le **seuil de classification utilisé est de `{optimal_threshold:.4f}`**. Ce seuil a été spécifiquement optimisé lors de l'entraînement pour maximiser la métrique F-beta.")
                st.markdown(f"- Si la probabilité calculée (`{prediction_proba:.4f}`) est supérieure ou égale au seuil (`{optimal_threshold:.4f}`), le client est classé comme 'Risque Élevé'.")

            except Exception as e:
                st.error(f"Une erreur est survenue lors de l'exécution de la prédiction : {e}")
                st.warning("Vérifiez que toutes les valeurs saisies sont numériques et cohérentes avec les attentes du modèle.")
                logger.exception("Erreur lors de la prédiction Streamlit:")

    else:
        st.error("L'application n'a pas pu être initialisée car le modèle ou ses métadonnées n'ont pas été chargés. Veuillez résoudre les erreurs signalées ci-dessus et relancer l'application.")

with tab2:
    st.header("Analyse du Data Drift (Evidently AI)")
    st.markdown("""
    Cette section affiche un rapport interactif généré par [Evidently AI](https://evidentlyai.com/)
    pour surveiller le **Data Drift** (dérive des données) entre vos données de référence (entraînement)
    et vos données actuelles (production). Un drift significatif peut indiquer que votre modèle
    doit être réentraîné.

    **Ce rapport est généré en dehors de l'application Streamlit via Docker.**
    """)
    # Appel de la fonction pour afficher le rapport Evidently
    # Le chemin du fichier est passé pour que la fonction puisse le charger.
    display_evidently_report(EVLIDENTLY_REPORT_PATH)

    st.markdown("---")
    st.subheader("Comment générer le rapport de Data Drift :")
    st.info(f"""
    1.  **Assurez-vous que Docker Desktop est installé et en cours d'exécution.**
    2.  **Ouvrez un terminal ou invite de commande** et naviguez jusqu'au dossier `monitoring` de votre projet :
        `cd C:\\Users\\jonjo\\Documents\\open classrooms\\Projet 7\\monitoring`
    3.  **Construisez l'image Docker** (une seule fois ou après modification du Dockerfile/requirements.txt) :
        `docker build -t evidently-report-generator .`
    4.  **Exécutez le conteneur Docker** pour générer le rapport HTML :
        `docker run --rm -v "C:\\Users\\jonjo\\Documents\\open classrooms\\Projet 7:/app" evidently-report-generator`

    Ce dernier commande va :
    * `docker run`: Exécuter un conteneur.
    * `--rm`: Supprimer le conteneur après son exécution (nettoyage automatique).
    * `-v "C:\\Users\\jonjo\\Documents\\open classrooms\\Projet 7:/app"`: **Monter le volume.** C'est crucial ! Cela lie votre dossier de projet local (là où se trouve `app.py` et où vous voulez le `evidently_data_drift_report.html`) au dossier `/app` à l'intérieur du conteneur. Ainsi, le rapport généré par `generate_report.py` dans `/app/evidently_data_drift_report.html` sera directement écrit dans votre dossier `C:\\Users\\jonjo\\Documents\\open classrooms\\Projet 7` sur votre machine.
    * `evidently-report-generator`: Le nom de l'image Docker que nous avons construite.

    Après avoir exécuté la commande `docker run`, le fichier `evidently_data_drift_report.html` devrait apparaître ou être mis à jour à la racine de votre dossier de projet (là où se trouve `app.py`). Vous pourrez alors actualiser votre page Streamlit pour voir le rapport.
    """)