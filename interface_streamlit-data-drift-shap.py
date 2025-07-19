import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap
import mlflow
import mlflow.pyfunc
import logging
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import io
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt # Added for waterfall plot
import json # Ensure json is imported for metadata loading

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Désactiver les avertissements Evidently s'ils sont trop nombreux
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="evidently")

# --- Configuration AWS S3 ---
AWS_S3_BUCKET_NAME = st.secrets["aws"]["bucket_name"]
AWS_ACCESS_KEY_ID = st.secrets["aws"]["aws_access_key_id"]
AWS_SECRET_ACCESS_KEY = st.secrets["aws"]["aws_secret_access_key"]

# Initialisation du client S3
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# --- Fonctions de chargement de données et de modèles ---

@st.cache_data(show_spinner="Chargement des données depuis S3...")
def load_data_from_s3(file_key):
    """Charge un fichier CSV depuis un bucket S3."""
    try:
        obj = s3_client.get_object(Bucket=AWS_S3_BUCKET_NAME, Key=file_key)
        data = pd.read_csv(io.BytesIO(obj['Body'].read()))
        logger.info(f"Fichier '{file_key}' chargé avec succès depuis S3.")
        return data
    except NoCredentialsError:
        st.error("Identifiants AWS non trouvés. Veuillez configurer les secrets Streamlit.")
        logger.error("AWS credentials not found.")
        return None
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            st.error(f"Le fichier '{file_key}' est introuvable dans le bucket S3.")
            logger.error(f"S3 file not found: {file_key}")
        else:
            st.error(f"Erreur lors du chargement depuis S3: {e}")
            logger.error(f"Error loading from S3: {e}")
        return None
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue lors du chargement des données depuis S3: {e}")
        logger.error(f"Unexpected error loading data from S3: {e}")
        return None

@st.cache_resource(show_spinner="Chargement des métadonnées du modèle depuis S3...")
def load_model_metadata_from_s3(file_key="model_metadata.json"):
    """Charge les métadonnées du modèle depuis un fichier JSON sur S3."""
    try:
        obj = s3_client.get_object(Bucket=AWS_S3_BUCKET_NAME, Key=file_key)
        metadata = json.loads(obj['Body'].read().decode('utf-8')) # Use json.loads for direct JSON parsing
        logger.info("Métadonnées du modèle chargées depuis S3.")
        return metadata
    except NoCredentialsError:
        st.error("Identifiants AWS non trouvés pour les métadonnées du modèle.")
        logger.error("AWS credentials not found for model metadata.")
        return None
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            st.warning(f"Le fichier de métadonnées '{file_key}' est introuvable dans le bucket S3. Utilisation des métadonnées par défaut si possible.")
            logger.warning(f"Model metadata file not found: {file_key}")
            return None
        else:
            st.error(f"Erreur lors du chargement des métadonnées depuis S3: {e}")
            logger.error(f"Error loading model metadata from S3: {e}")
        return None
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue lors du chargement des métadonnées du modèle depuis S3: {e}")
        logger.error(f"Unexpected error loading model metadata from S3: {e}")
        return None

# --- Chargement local des ressources MLflow ---

@st.cache_resource(show_spinner="Chargement du pipeline MLflow localement...")
def load_mlflow_pipeline_local(model_local_dir="./downloaded_assets/modele_mlflow"):
    """
    Charge un pipeline MLflow depuis un répertoire local.
    Assurez-vous que le modèle MLflow est bien présent dans ce répertoire.
    """
    if not os.path.exists(model_local_dir) or not os.listdir(model_local_dir): # Check if directory is empty
        # Tenter de télécharger si le répertoire n'existe pas localement ou est vide
        st.warning(f"Répertoire du modèle MLflow '{model_local_dir}' non trouvé ou vide localement. Tentative de téléchargement depuis S3...")
        try:
            download_mlflow_model_from_s3("modele_mlflow", model_local_dir)
        except Exception as e:
            st.error(f"Échec du téléchargement du modèle MLflow depuis S3 : {e}")
            return None

    try:
        pipeline = mlflow.pyfunc.load_model(model_uri=model_local_dir)
        logger.info("Streamlit: Pipeline chargé depuis './downloaded_assets/modele_mlflow'.")
        return pipeline
    except Exception as e:
        logger.error(f"Erreur lors du chargement du pipeline MLflow: {e}", exc_info=True)
        st.error(f"Erreur lors du chargement du pipeline MLflow : {e}. Assurez-vous que le modèle est compatible avec les dépendances installées.")
        return None

@st.cache_resource(show_spinner="Téléchargement du modèle MLflow depuis S3...")
def download_mlflow_model_from_s3(s3_prefix, local_dir):
    """Télécharge un modèle MLflow complet (répertoire) depuis S3."""
    # This function is now called by load_mlflow_pipeline_local if needed
    # We should ensure it doesn't try to re-download if already present
    if os.path.exists(local_dir) and os.listdir(local_dir):
        logger.info(f"Le répertoire local '{local_dir}' existe déjà et n'est pas vide, pas de téléchargement nécessaire.")
        return

    os.makedirs(local_dir, exist_ok=True)
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=AWS_S3_BUCKET_NAME, Prefix=s3_prefix)

        files_downloaded_count = 0
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    object_key = obj["Key"]
                    # Créer la structure de répertoire locale
                    local_file_path = os.path.join(local_dir, os.path.relpath(object_key, s3_prefix))
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                    if not object_key.endswith('/'): # Ignorer les 'dossiers' S3
                        s3_client.download_file(AWS_S3_BUCKET_NAME, object_key, local_file_path)
                        logger.info(f"Téléchargé : {object_key} vers {local_file_path}")
                        files_downloaded_count += 1
            else:
                st.warning(f"Aucun objet trouvé sous le préfixe S3: {s3_prefix}")
                logger.warning(f"No objects found under S3 prefix: {s3_prefix}")
                return # Sortir si rien n'est trouvé
        
        if files_downloaded_count > 0:
            logger.info(f"Modèle MLflow téléchargé avec succès depuis S3 '{s3_prefix}' vers '{local_dir}' ({files_downloaded_count} fichiers).")
        else:
            st.warning(f"Aucun fichier téléchargé pour le modèle MLflow sous le préfixe '{s3_prefix}'. Vérifiez le chemin S3.")

    except NoCredentialsError:
        st.error("Identifiants AWS non trouvés pour le téléchargement du modèle MLflow.")
        logger.error("AWS credentials not found for MLflow model download.")
        raise
    except ClientError as e:
        st.error(f"Erreur Client S3 lors du téléchargement du modèle MLflow: {e}")
        logger.error(f"S3 Client Error during MLflow model download: {e}")
        raise
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue lors du téléchargement du modèle MLflow: {e}")
        logger.error(f"Unexpected error during MLflow model download: {e}")
        raise


# --- Préparation des données d'entraînement (pour SHAP et Data Drift) ---

@st.cache_data(show_spinner="Chargement des données d'entraînement pour SHAP et Data Drift depuis S3...")
def load_training_data_for_shap(file_key="input/application_train.csv"):
    """Charge les données d'entraînement complètes pour les calculs SHAP et Data Drift."""
    try:
        data = load_data_from_s3(file_key)
        if data is not None:
            # Assurez-vous que TARGET est géré si nécessaire
            if 'TARGET' in data.columns:
                data = data.drop(columns=['TARGET'], errors='ignore')
            logger.info("Données d'entraînement pour SHAP et Data Drift chargées.")
            return data
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement des données d'entraînement pour SHAP: {e}")
        logger.error(f"Error loading training data for SHAP: {e}")
        return None

# --- Ingénierie des Caractéristiques pour les données de référence SHAP (simplifié) ---

@st.cache_data(show_spinner="Préparation des données pour l'ingénierie des caractéristiques...")
def run_feature_engineering_pipeline(num_rows=None):
    """
    Simule une partie de l'ingénierie des caractéristiques pour obtenir des données brutes
    à utiliser comme base pour SHAP et le préprocesseur.
    """
    raw_data = load_training_data_for_shap(file_key="input/application_train.csv") 
    if raw_data is None:
        return None
    
    if num_rows:
        return raw_data.sample(min(num_rows, len(raw_data)), random_state=42)
    return raw_data

# --- Explication SHAP ---

class IdentityPreprocessor:
    """Un préprocesseur factice qui ne fait rien. Utilisé quand aucun préprocesseur explicite n'est trouvé."""
    def transform(self, X):
        return X
    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        # Fallback pour les cas où input_features n'est pas fourni et la transformation ne modifie pas les noms
        return [f"col_{i}" for i in range(X.shape[1])] if hasattr(X, 'shape') else []


@st.cache_resource(show_spinner="Calcul de l'explainer SHAP...")
def load_shap_explainer(_pyfunc_pipeline, all_training_features):
    """
    Charge l'explainer SHAP et le préprocesseur à partir du pipeline MLflow.
    Args:
        _pyfunc_pipeline: L'objet PyFuncModel chargé par MLflow.
        all_training_features: Liste de toutes les features d'entraînement attendues.
    Returns:
        tuple: (shap.Explainer, preprocessor) ou (None, None) en cas d'échec.
    """
    sklearn_model_or_pipeline = None
    if hasattr(_pyfunc_pipeline, '_model_impl'):
        sklearn_model_or_pipeline = _pyfunc_pipeline._model_impl
    
    if sklearn_model_or_pipeline is None:
        st.error("Impossible d'extraire le modèle/pipeline scikit-learn du modèle MLflow. L'explainer SHAP ne peut pas être initialisé correctement.")
        logger.error("Could not extract a scikit-learn model or Pipeline from the MLflow PyFuncModel.")
        return None, None

    preprocessor = None
    final_model = None

    if isinstance(sklearn_model_or_pipeline, Pipeline) or hasattr(sklearn_model_or_pipeline, 'named_steps'):
        logger.info("Modèle MLflow chargé est reconnu comme un Pipeline scikit-learn.")
        # Correction d'une faute de frappe ici : 'sklearn_model_or_or_pipeline' -> 'sklearn_model_or_pipeline'
        if 'preprocessor' in sklearn_model_or_pipeline.named_steps: 
            preprocessor = sklearn_model_or_pipeline.named_steps['preprocessor']
            logger.info("Préprocesseur nommé 'preprocessor' trouvé dans le pipeline.")
        else:
            logger.warning("Le pipeline scikit-learn ne contient pas d'étape nommée 'preprocessor'. Utilisation d'IdentityPreprocessor.")
            preprocessor = IdentityPreprocessor()
        
        final_model = sklearn_model_or_pipeline.steps[-1][1] 
        logger.info(f"Modèle final extrait du pipeline: {type(final_model)}")
    else:
        logger.info("Le modèle MLflow chargé n'est pas un pipeline scikit-learn avec 'named_steps'. Traitement comme un modèle final direct.")
        final_model = sklearn_model_or_pipeline
        preprocessor = IdentityPreprocessor()

    if final_model is None:
        st.error("Impossible d'identifier le modèle final pour l'explainer SHAP.")
        logger.error("Final model could not be extracted or identified.")
        return None, None

    ref_data_raw = run_feature_engineering_pipeline(num_rows=1000)
    if ref_data_raw is None:
        st.error("Impossible de charger les données de référence pour l'explainer SHAP.")
        return None, None

    # S'assurer que seules les colonnes d'entraînement sont utilisées
    ref_data_raw_filtered = ref_data_raw[all_training_features]
    
    ref_data_processed = preprocessor.transform(ref_data_raw_filtered)
    
    processed_feature_names = all_training_features
    try:
        if hasattr(preprocessor, 'get_feature_names_out') and callable(preprocessor.get_feature_names_out):
            try:
                processed_feature_names = preprocessor.get_feature_names_out(input_features=all_training_features)
            except TypeError:
                processed_feature_names = preprocessor.get_feature_names_out()
        elif hasattr(ref_data_processed, 'columns'):
            processed_feature_names = ref_data_processed.columns.tolist()
        else:
            processed_feature_names = [f"col_{i}" for i in range(ref_data_processed.shape[1])]
            logger.warning("Impossible d'obtenir les noms de features du préprocesseur. Noms génériques utilisés.")
    except Exception as e:
        logger.warning(f"Erreur lors de l'appel de get_feature_names_out ou extraction des colonnes: {e}. Noms génériques utilisés.")
        processed_feature_names = [f"col_{i}" for i in range(ref_data_processed.shape[1])]


    ref_data_df = pd.DataFrame(ref_data_processed, columns=processed_feature_names)
    
    explainer = shap.Explainer(final_model, ref_data_df)
    return explainer, preprocessor


# --- Fonctions d'affichage ---

def plot_feature_importance(explainer, shap_values, feature_names, top_n=10):
    """Affiche les importances globales des caractéristiques SHAP."""
    if explainer is None or shap_values is None or not feature_names:
        st.warning("Impossible d'afficher l'importance des caractéristiques : données SHAP manquantes.")
        return

    if isinstance(shap_values, list):
        shap_values_abs_mean = np.abs(np.array(shap_values[0])).mean(0)
    else:
        shap_values_abs_mean = np.abs(np.array(shap_values)).mean(0)

    if len(shap_values_abs_mean) != len(feature_names):
        st.error(f"Incohérence entre les valeurs SHAP ({len(shap_values_abs_mean)}) et les noms de features ({len(feature_names)}).")
        return

    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Importance': shap_values_abs_mean
    }).sort_values(by='SHAP_Importance', ascending=False)

    fig = px.bar(
        df_importance.head(top_n),
        x='SHAP_Importance',
        y='Feature',
        orientation='h',
        title=f'Top {top_n} Importances des Caractéristiques (Moyenne Absolue des Valeurs SHAP)',
        labels={'SHAP_Importance': 'Importance SHAP (Moyenne Absolue)', 'Feature': 'Caractéristique'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def plot_individual_explanation(explainer, shap_values_individual, processed_features_df, feature_names, client_id):
    """Affiche l'explication SHAP pour un client individuel."""
    if explainer is None or shap_values_individual is None or processed_features_df is None:
        st.warning("Impossible d'afficher l'explication individuelle : données SHAP manquantes.")
        return
    
    if isinstance(shap_values_individual, list):
        shap_values_individual = shap_values_individual[0] 

    try:
        expected_value = explainer.expected_value
        
        if not isinstance(processed_features_df, pd.DataFrame):
            processed_features_df = pd.DataFrame([processed_features_df], columns=feature_names)
            
        shap.initjs()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_individual, 
                base_values=expected_value, 
                data=processed_features_df.iloc[0].values, 
                feature_names=feature_names
            ),
            max_display=20,
            show=False,
            ax=ax
        )
        ax.set_title(f"Explication SHAP pour le client {client_id}")
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Erreur lors de l'affichage de l'explication SHAP individuelle : {e}")
        logger.error(f"Error plotting individual SHAP explanation: {e}", exc_info=True)


# --- Fonctions principales de l'application Streamlit ---

st.set_page_config(layout="wide", page_title="Prédiction de Risque de Crédit et Explicabilité")

st.title("Tableau de Bord de Prédiction de Risque de Crédit")
st.write("Cette application prédit le risque de défaut de paiement pour les demandes de crédit et fournit des explications sur les prédictions.")

# Chargement des données et du modèle
data = load_data_from_s3("input/application_test.csv") 
if data is None:
    st.stop()

# Chargement des métadonnées du modèle
model_metadata = load_model_metadata_from_s3(file_key="model_metadata.json") 
if model_metadata is None:
    st.error("Impossible de charger les métadonnées du modèle. Certaines fonctionnalités pourraient être limitées.")
    all_training_features = data.columns.tolist() # Fallback
    threshold = 0.5 # Valeur par défaut
else:
    all_training_features = model_metadata['training_features'] if 'training_features' in model_metadata else data.columns.tolist()
    threshold = model_metadata.get('threshold', 0.5)

# --- NOUVELLE LOGIQUE : Assurez-vous que 'SK_ID_CURR' n'est JAMAIS dans les features d'entraînement ---
# Cette vérification est maintenant effectuée quelle que soit la source de all_training_features.
if 'SK_ID_CURR' in all_training_features:
    all_training_features.remove('SK_ID_CURR')
    logger.info("SK_ID_CURR retiré de all_training_features.")
    
# Assurez-vous que les colonnes nécessaires sont présentes dans 'data'
missing_features_in_data = [f for f in all_training_features if f not in data.columns]
if missing_features_in_data:
    st.warning(f"Attention : Les caractéristiques suivantes du modèle sont manquantes dans les données de test : {', '.join(missing_features_in_data)}. Le modèle pourrait ne pas fonctionner comme prévu.")
    all_training_features = [f for f in all_training_features if f in data.columns]


# Chargement du pipeline MLflow
pipeline = load_mlflow_pipeline_local()
if pipeline is None:
    st.stop()

# --- Sidebar pour la sélection du client ---
st.sidebar.header("Sélection du Client")
client_ids = data['SK_ID_CURR'].tolist()
selected_client_id = st.sidebar.selectbox("Sélectionnez un ID Client :", client_ids)

# Trouver les données du client sélectionné
client_data_raw = data[data['SK_ID_CURR'] == selected_client_id].drop(columns=['SK_ID_CURR']).iloc[0]
client_data_df_input = pd.DataFrame([client_data_raw]) # Pour l'entrée du pipeline

# Log pour le débogage :
logger.info(f"Features passées au pipeline: {all_training_features}")
logger.info(f"Colonnes disponibles dans client_data_df_input: {client_data_df_input.columns.tolist()}")

# Prétraiter les données du client pour la prédiction
client_data_filtered_for_pipeline = client_data_df_input[all_training_features]

# --- Prédiction ---
with st.spinner("Calcul de la prédiction..."):
    try:
        prediction_proba = pipeline.predict(client_data_filtered_for_pipeline)[0]
        prediction = (prediction_proba >= threshold).astype(int)
        
        st.subheader("Résultat de la Prédiction")
        col_pred, col_proba = st.columns(2)
        with col_pred:
            if prediction == 1:
                st.error(f"**Prédiction : Risque Élevé de Défaut (Crédit Refusé)**")
            else:
                st.success(f"**Prédiction : Faible Risque de Défaut (Crédit Accordé)**")
        with col_proba:
            st.info(f"Probabilité de défaut : **{prediction_proba:.2f}**")
            st.info(f"Seuil de décision : **{threshold:.2f}**")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction pour le client : {e}")
        logger.error(f"Prediction error for client {selected_client_id}: {e}", exc_info=True)
        prediction_proba = None
        prediction = None

st.write("---")

# --- Explication SHAP ---
st.subheader("Explication SHAP de la Prédiction")

# Charger l'explainer SHAP et le préprocesseur
explainer, preprocessor = load_shap_explainer(pipeline, all_training_features)

if explainer is not None and preprocessor is not None:
    try:
        # Prétraiter les données du client pour SHAP avec le préprocesseur extrait
        client_data_processed_for_shap = preprocessor.transform(client_data_filtered_for_pipeline)

        # Récupérer les noms des features après prétraitement si le préprocesseur le permet
        if hasattr(preprocessor, 'get_feature_names_out') and callable(preprocessor.get_feature_names_out):
            try:
                processed_feature_names = preprocessor.get_feature_names_out(input_features=all_training_features)
            except TypeError:
                processed_feature_names = preprocessor.get_feature_names_out()
        else:
            processed_feature_names = [f"col_{i}" for i in range(client_data_processed_for_shap.shape[1])]
            logger.warning("Impossible d'obtenir les noms de features du préprocesseur pour SHAP. Noms génériques utilisés.")

        # Calculer les valeurs SHAP
        with st.spinner("Calcul des valeurs SHAP..."):
            shap_values = explainer.shap_values(client_data_processed_for_shap)
        
        st.write("Les valeurs SHAP montrent l'impact de chaque caractéristique sur la prédiction du modèle.")
        
        # Affichage de l'explication individuelle
        if prediction_proba is not None:
            if isinstance(shap_values, list) and len(shap_values) > 1:
                individual_shap_values = shap_values[1][0]
            else:
                individual_shap_values = shap_values[0]
            
            client_processed_df = pd.DataFrame(client_data_processed_for_shap, columns=processed_feature_names)

            plot_individual_explanation(explainer, individual_shap_values, client_processed_df, processed_feature_names, selected_client_id)
            
            st.subheader("Importance Globale des Caractéristiques (Basée sur l'échantillon SHAP)")
            plot_feature_importance(explainer, shap_values, processed_feature_names)

    except Exception as e:
        st.error(f"Une erreur est survenue lors du calcul ou de l'affichage des valeurs SHAP : {e}")
        logger.error(f"SHAP calculation/plotting error: {e}", exc_info=True)
else:
    st.warning("L'explainer SHAP n'a pas pu être initialisé. Les explications ne sont pas disponibles.")

st.write("---")

# --- Analyse de la Dérive des Données (Data Drift) ---
st.subheader("Analyse de la Dérive des Données (Data Drift)")
st.warning("Cette section est en cours de développement et nécessiterait une intégration avec des outils comme Evidently AI ou NannyML pour une analyse complète de la dérive des données.")
st.info("Pour une implémentation complète, il faudrait charger un dataset de référence (entraînement) et le comparer avec les données de production (ou ici, les données de test).")
