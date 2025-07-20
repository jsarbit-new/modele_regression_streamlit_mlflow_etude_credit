import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import json
import logging
import os
import streamlit.components.v1 as components

# Imports Evidently (version 0.2.8, mais fonctionne aussi avec d'autres versions récentes)
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab

# --- 1. Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="Prédiction de Défaut Client & Surveillance",
    page_icon="📊",
    layout="wide"
)

# --- 2. Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 3. Configuration MLflow ---
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
    """Charge le modèle et ses métadonnées depuis MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        model_versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")

        if not model_versions:
            error_msg = f"Aucune version du modèle '{MLFLOW_MODEL_NAME}' trouvée."
            logger.error(f"Streamlit: {error_msg}")
            raise ValueError(error_msg)

        latest_version = max(model_versions, key=lambda mv: mv.version)
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/{latest_version.version}"
        model = mlflow.sklearn.load_model(model_uri=model_uri)
        logger.info(f"Streamlit: Modèle '{MLFLOW_MODEL_NAME}' (v{latest_version.version}) chargé.")

        optimal_threshold = None
        features_info = None
        all_training_features = None

        model_metadata_tags = latest_version.tags if hasattr(latest_version, 'tags') else {}
        if 'mlflow.log_model.metadata.optimal_threshold' in model_metadata_tags:
            optimal_threshold = float(model_metadata_tags['mlflow.log_model.metadata.optimal_threshold'])
        if 'mlflow.log_model.metadata.features_info_for_streamlit' in model_metadata_tags:
            features_info = json.loads(model_metadata_tags['mlflow.log_model.metadata.features_info_for_streamlit'])
        if 'mlflow.log_model.metadata.all_training_features' in model_metadata_tags:
            all_training_features = json.loads(model_metadata_tags['mlflow.log_model.metadata.all_training_features'])

        if not all([optimal_threshold, features_info, all_training_features]):
            logger.warning("Certaines métadonnées sont manquantes dans les tags. Tentative de récupération depuis les paramètres de la run...")
            run_info = client.get_run(latest_version.run_id)
            run_info_params = run_info.data.params
            
            if optimal_threshold is None and 'optimal_threshold_value' in run_info_params:
                optimal_threshold = float(run_info_params['optimal_threshold_value'])
            
            if features_info is None and 'features_info_for_streamlit_json' in run_info_params:
                features_info = json.loads(run_info_params['features_info_for_streamlit_json'])
            
            if all_training_features is None and 'all_training_features_names' in run_info_params:
                all_training_features = json.loads(run_info_params['all_training_features_names'])

        if not features_info:
            raise ValueError("Impossible de charger les features d'entrée.")
        
        if not optimal_threshold:
            optimal_threshold = 0.5
            logger.warning(f"Seuil optimal non trouvé. Utilisation de la valeur par défaut: {optimal_threshold}.")
        
        if not all_training_features:
            logger.warning("Liste complète des features d'entraînement non trouvée. Utilisation des features importantes en fallback.")
            all_training_features = list(features_info.keys())

        return model, features_info, optimal_threshold, all_training_features

    except Exception as e:
        st.error(f"Échec critique lors du chargement du modèle ou de ses métadonnées: {e}")
        st.info("""
            **Vérifiez les points suivants pour résoudre le problème :**
            1.  **Serveur MLflow UI démarré.**
            2.  **Modèle enregistré** dans l'onglet "Models" avec le nom `HomeCreditLogisticRegressionPipeline`.
            3.  **Logs d'entraînement** pour confirmer que les métadonnées ont été correctement logguées.
        """)
        st.stop()
        return None, None, None, None

# --- Chargement des données de référence (entraînement) ---
@st.cache_resource(show_spinner="Chargement des données de référence pour l'analyse de dérive...")
def load_reference_data():
    """
    Charge les données de référence (entraînement).
    """
    try:
        reference_df = pd.read_csv("C:\\Users\\jonjo\\Documents\\open classrooms\\Projet 7\\input\\application_train.csv")
        logger.info(f"Données de référence chargées avec succès. Nombre d'échantillons: {len(reference_df)}")
        return reference_df
    except FileNotFoundError:
        st.error("Fichier de données de référence non trouvé. Veuillez vérifier le chemin.")
        st.stop()
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement des données de référence : {e}")
        st.stop()
        return None

# --- Fonction de génération et d'affichage du rapport ---
def generate_and_display_evidently_report(reference_df, current_df):
    """
    Génère un rapport de dérive des données et l'affiche dans Streamlit.
    (Syntaxe pour Evidently 0.2.8)
    """
    try:
        st.info("Génération du rapport en cours. Cela peut prendre quelques instants...")

        # Échantillonnage de 20% des données pour des raisons de performance
        sample_frac = 0.2
        ref_sample = reference_df.sample(frac=sample_frac, random_state=42)
        current_sample = current_df.sample(frac=sample_frac, random_state=42)
        
        st.info(f"Génération du rapport sur un échantillon de {len(ref_sample)} lignes de référence et {len(current_sample)} lignes actuelles.")

        report_file_path = "evidently_data_drift_report_temp.html"
        
        data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
        data_drift_dashboard.calculate(reference_data=ref_sample, current_data=current_sample)
        data_drift_dashboard.save(report_file_path)
        
        with open(report_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        components.html(html_content, height=1000, scrolling=True)
        st.success("Rapport Evidently généré et affiché avec succès.")
        
        os.remove(report_file_path)
    
    except Exception as e:
        st.error(f"Erreur lors de la génération ou de l'affichage du rapport Evidently : {e}")
        logger.exception("Erreur lors de l'exécution du rapport Evidently dans Streamlit:")

# --- 5. Chargement des Ressources au Démarrage ---
model, features_info, optimal_threshold, all_training_features = load_mlflow_model_and_metadata()
reference_data = load_reference_data()

# --- 6. Contenu Principal de la Page Streamlit ---
st.title("📊 Prédiction de Défaut Client & Surveillance du Modèle")

tab1, tab2 = st.tabs(["Prédiction de Prêt", "Analyse du Data Drift"])

with tab1:
    st.markdown("""
    Cette application vous permet de simuler une prédiction de risque de défaut pour un client.
    """)
    if model and features_info:
        st.sidebar.header("Informations sur le Modèle")
        st.sidebar.write(f"**Nom du Modèle :** `{MLFLOW_MODEL_NAME}`")
        try:
            client = mlflow.tracking.MlflowClient()
            current_version_obj = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=['None', 'Production', 'Staging'])
            current_version = current_version_obj[0].version if current_version_obj else "N/A"
            st.sidebar.write(f"**Version du Modèle :** `{current_version}`")
        except Exception:
            st.sidebar.warning("Impossible de récupérer la version du modèle.")
        st.sidebar.write(f"**Seuil Optimal Utilisé :** `{optimal_threshold:.4f}`")
        st.sidebar.write(f"**Nombre de Caractéristiques Importantes :** `{len(features_info)}`")

        st.subheader("Saisie des Caractéristiques Client Importantes")
        user_inputs = {}
        num_columns = 2
        cols = st.columns(num_columns)
        
        for i, (feature_original_name, info) in enumerate(features_info.items()):
            display_name = info.get("display_name", feature_original_name)
            min_val_hint = info.get("min_val")
            max_val_hint = info.get("max_val")
            default_value = (float(min_val_hint) + float(max_val_hint)) / 2 if min_val_hint is not None and max_val_hint is not None else 0.0
            with cols[i % num_columns]:
                st.write(f"**{display_name}**")
                st.caption(f"Plage: [{min_val_hint:.2f} - {max_val_hint:.2f}]" if min_val_hint is not None and max_val_hint is not None else "Plage: N/A")
                user_inputs[feature_original_name] = st.number_input(
                    label=" ",
                    value=float(default_value),
                    min_value=float(min_val_hint) if min_val_hint is not None else None,
                    max_value=float(max_val_hint) if max_val_hint is not None else None,
                    format="%.4f",
                    key=f"input_{feature_original_name}"
                )
        st.markdown("---")
        if st.button("Obtenir la Prédiction", help="Cliquez pour exécuter le modèle avec les valeurs saisies."):
            model_input_data = {
                feature_name: user_inputs.get(feature_name, 0.0)
                for feature_name in all_training_features
            }
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
            except Exception as e:
                st.error(f"Une erreur est survenue lors de l'exécution de la prédiction : {e}")
                logger.exception("Erreur lors de la prédiction Streamlit:")
    else:
        st.error("L'application n'a pas pu être initialisée.")

with tab2:
    st.header("Analyse du Data Drift (Evidently AI)")
    st.markdown("""
    Cette section génère et affiche un rapport de **Data Drift** directement dans l'application.
    Le rapport compare les données d'entraînement (référence) aux données de production simulées.
    """)
    
    if st.button("Générer et afficher le rapport de Data Drift"):
        st.info("Génération du rapport en cours. Cela peut prendre quelques instants...")
        
        # --- Simulation de données de production ---
        # Remplacement du bloc par la lecture de vos données de production réelles
        df_production = reference_data.copy()
        
        # Simulation d'un drift sur certaines colonnes
        if 'AMT_CREDIT' in df_production.columns:
            df_production['AMT_CREDIT'] = df_production['AMT_CREDIT'] * np.random.normal(1.2, 0.1, len(df_production))
        if 'DAYS_BIRTH' in df_production.columns:
            df_production['DAYS_BIRTH'] = df_production['DAYS_BIRTH'] + np.random.randint(-365, 365, len(df_production))
        
        # Appel de la fonction de génération et d'affichage
        generate_and_display_evidently_report(reference_data, df_production)
    else:
        st.warning("Cliquez sur le bouton pour générer le rapport de dérive des données.")