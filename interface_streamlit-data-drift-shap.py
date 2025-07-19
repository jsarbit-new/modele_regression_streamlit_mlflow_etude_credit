import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import json
import logging
import os
import streamlit.components.v1 as components
import shap
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab
from sklearn.model_selection import train_test_split
import boto3 # NEW: Import boto3 for AWS S3 interaction
from botocore.exceptions import NoCredentialsError # NEW: Import for handling AWS credentials errors

# --- 1. Page Configuration Streamlit ---
st.set_page_config(
    page_title="Prédiction de Défaut Client & Surveillance",
    page_icon="📊",
    layout="wide"
)

# --- 2. Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 3. AWS S3 Configuration ---
# REMPLACEZ par le nom EXACT de votre bucket S3
S3_BUCKET_NAME = "modele-regression-streamlit-mlflow-etude-credit"
# REMPLACEZ par la région de votre bucket S3 (ex: "eu-west-3" pour Paris)
AWS_REGION = "eu-west-3" # Assurez-vous que c'est la bonne région de votre bucket S3

# Chemin local temporaire où les fichiers seront téléchargés dans l'environnement Streamlit Cloud
LOCAL_DOWNLOAD_DIR = "./downloaded_assets"
os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True) # Crée le dossier local si nécessaire

# --- S3 Keys for specific files/directories ---
# REMPLACEZ par le chemin réel de votre dataset sur S3
DATASET_S3_KEY = "input/application_train.csv"
# REMPLACEZ par le préfixe du dossier de votre modèle MLflow sur S3 (doit se terminer par '/')
# Ex: si votre modèle MLflow est dans s3://your-bucket/modele_mlflow/
# alors MODEL_S3_KEY_PREFIX = "modele_mlflow/"
MODEL_S3_KEY_PREFIX = "modele_mlflow/"

# --- NEW: Function to download a single file from S3 (cached) ---
@st.cache_resource
def download_file_from_s3(s3_key, local_path):
    """
    Télécharge un fichier unique depuis S3 vers un chemin local.
    Les identifiants AWS sont récupérés des secrets Streamlit Cloud.
    """
    try:
        aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        if not aws_access_key_id or not aws_secret_access_key:
            st.error("Erreur : Les identifiants AWS (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) ne sont pas configurés dans les secrets Streamlit.")
            return None

        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=AWS_REGION
        )

        st.info(f"Téléchargement de '{s3_key}' depuis S3 (bucket: '{S3_BUCKET_NAME}')...")
        s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
        st.success(f"Fichier '{s3_key}' téléchargé avec succès vers '{local_path}'!")
        return local_path
    except NoCredentialsError:
        st.error("Erreur d'authentification AWS. Vérifiez vos identifiants AWS dans les secrets Streamlit Cloud.")
        return None
    except Exception as e:
        st.error(f"Erreur lors du téléchargement de '{s3_key}' depuis S3 : {e}")
        logger.exception(f"Erreur lors du téléchargement de '{s3_key}' depuis S3:")
        return None

# --- NEW: Function to download an entire directory (prefix) from S3 (cached) ---
@st.cache_resource
def download_s3_directory(s3_prefix, local_dir):
    """
    Télécharge tous les objets sous un préfixe S3 donné vers un répertoire local.
    Utile pour les dossiers de modèles MLflow.
    """
    try:
        aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        if not aws_access_key_id or not aws_secret_access_key:
            st.error("Erreur : Les identifiants AWS ne sont pas configurés pour le téléchargement de dossier.")
            return False

        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=AWS_REGION
        )

        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=s3_prefix)

        os.makedirs(local_dir, exist_ok=True)

        st.info(f"Téléchargement du dossier '{s3_prefix}' depuis S3 (bucket: '{S3_BUCKET_NAME}')...")
        files_downloaded = 0
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    s3_key = obj["Key"]
                    # Skip directories themselves if they appear as objects
                    if s3_key.endswith('/'):
                        continue
                    
                    # Construct local path, preserving subdirectories
                    relative_path = os.path.relpath(s3_key, s3_prefix)
                    local_path = os.path.join(local_dir, relative_path)
                    
                    # Ensure local directory exists for the file
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
                    # Only download if the file doesn't exist or is different (optional, for robustness)
                    # For simplicity, we re-download every time for @st.cache_resource
                    s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
                    files_downloaded += 1
        
        if files_downloaded > 0:
            st.success(f"Dossier '{s3_prefix}' téléchargé avec succès vers '{local_dir}' ({files_downloaded} fichiers)!")
            return True
        else:
            st.warning(f"Aucun fichier trouvé sous le préfixe '{s3_prefix}' dans le bucket '{S3_BUCKET_NAME}'.")
            return False

    except NoCredentialsError:
        st.error("Erreur d'authentification AWS. Vérifiez vos identifiants AWS dans les secrets Streamlit Cloud.")
        return False
    except Exception as e:
        st.error(f"Erreur lors du téléchargement du dossier '{s3_prefix}' depuis S3 : {e}")
        logger.exception(f"Erreur lors du téléchargement du dossier '{s3_prefix}' depuis S3:")
        return False


# --- 4. Feature Engineering Functions (Stubs) ---
# Dictionnaire des variables importantes à afficher dans l'interface
SHAP_IMPORTANT_FEATURES_INFO = {
    "AMT_CREDIT": {"display_name": "Montant du Prêt", "min_val": 50000.0, "max_val": 2000000.0},
    "AMT_ANNUITY": {"display_name": "Montant Annuité", "min_val": 1000.0, "max_val": 100000.0},
    "app_feature_15": {"display_name": "Ratio Crédit/Annuité", "min_val": 0.01, "max_val": 10.0},
    "app_feature_33": {"display_name": "Ancienneté Emploi (années)", "min_val": 0.0, "max_val": 50.0},
    "app_feature_21": {"display_name": "Taux Population Région", "min_val": 0.001, "max_val": 0.1},
    "app_feature_19": {"display_name": "Source Extérieure 1", "min_val": 0.0, "max_val": 1.0},
    "app_feature_31": {"display_name": "Source Extérieure 2", "min_val": 0.0, "max_val": 1.0},
    "app_feature_24": {"display_name": "Source Extérieure 3", "min_val": 0.0, "max_val": 1.0},
    "app_feature_45": {"display_name": "Nombre Enfants", "min_val": 0.0, "max_val": 10.0},
    "app_feature_17": {"display_name": "Age Client (années)", "min_val": 18.0, "max_val": 70.0},
}

# NOUVEAU: Dictionnaire de noms descriptifs pour TOUTES les variables
FULL_DESCRIPTIVE_NAMES = {
    # Variables ajoutées dans l'interface
    "AMT_CREDIT": "Montant du Prêt",
    "AMT_ANNUITY": "Montant Annuité",
    
    # Variables de la famille 'app_feature'
    "app_feature_15": "Ratio Crédit/Annuité",
    "app_feature_33": "Ancienneté Emploi (années)",
    "app_feature_21": "Taux Population Région",
    "app_feature_19": "Source Extérieure 1",
    "app_feature_31": "Source Extérieure 2",
    "app_feature_24": "Source Extérieure 3",
    "app_feature_45": "Nombre Enfants",
    "app_feature_17": "Age Client (années)",
    "app_feature_0": "Statut de la demande",
    "app_feature_1": "Statut de propriété",
    "app_feature_2": "Montant du bien",
    "app_feature_3": "Type de logement",
    "app_feature_4": "Type de famille",
    "app_feature_5": "Nb de jours depuis l'enregistrement",
    "app_feature_6": "Score 1 du client",
    "app_feature_7": "Score 2 du client",
    "app_feature_8": "Score 3 du client",
    "app_feature_9": "Nb d'enquêtes récentes",
    "app_feature_10": "Dernier changement d'ID",
    "app_feature_11": "Dernier changement de document",
    "app_feature_12": "Score financier 1",
    "app_feature_13": "Score financier 2",
    "app_feature_14": "Score financier 3",
    "app_feature_18": "Ratio Annuité/Revenu",
    "app_feature_20": "Type de paiement",
    "app_feature_22": "Score de crédit Bureau",
    "app_feature_23": "Nb de paiements manqués",
    "app_feature_25": "Ratio dette/revenu",
    "app_feature_26": "Nb de crédits en cours",
    "app_feature_27": "Nb de demandes par téléphone",
    "app_feature_28": "Nb de crédits renouvelables",
    "app_feature_29": "Nb de crédits soldés",
    "app_feature_30": "Montant de l'assurance",
    "app_feature_32": "Nb de jours depuis le dernier crédit",
    "app_feature_34": "Dernier changement de contact",
    "app_feature_35": "Dernière mise à jour d'info",
    "app_feature_36": "Montant des pénalités",
    "app_feature_37": "Montant des arriérés",
    "app_feature_38": "Nb de jours depuis le dernier contact",
    "app_feature_39": "Montant des paiements réguliers",
    "app_feature_40": "Ratio paiements/solde",
    "app_feature_41": "Nb de jours depuis le dernier paiement",
    "app_feature_42": "Dernier montant remboursé",
    "app_feature_43": "Nb de jours depuis le début du prêt",
    "app_feature_44": "Nb de paiements totaux",
    "app_feature_46": "Nb de paiements manqués totaux",
    "app_feature_47": "Montant total de la dette",
    "app_feature_48": "Ancienneté du crédit bureau",
    "app_feature_49": "Ratio crédit/revenu",
    
    # Variables des autres sources de données
    "bureau_feat_0": "Crédits Bureau",
    "bureau_feat_1": "Durée des crédits Bureau",
    "bureau_feat_2": "Ancienneté des crédits Bureau",
    "bureau_feat_3": "Dettes Bureau",
    "bureau_feat_4": "Crédits en cours Bureau",
    "prev_app_feat_0": "Ancienneté demandes précédentes",
    "prev_app_feat_1": "Taux d'acceptation demandes précédentes",
    "prev_app_feat_2": "Montant moyen demandes précédentes",
    "prev_app_feat_3": "Durée moyenne demandes précédentes",
    "prev_app_feat_4": "Ratio de remboursement demandes précédentes",
    "pos_feat_0": "Ancienneté POS Cash",
    "pos_feat_1": "Nb de paiements POS Cash",
    "pos_feat_2": "Montant POS Cash",
    "pos_feat_3": "Jours de retard POS Cash",
    "pos_feat_4": "Statut de paiement POS Cash",
    "install_feat_0": "Ancienneté Paiements acomptes",
    "install_feat_1": "Nb de paiements acomptes",
    "install_feat_2": "Montant paiements acomptes",
    "install_feat_3": "Paiements en retard acomptes",
    "install_feat_4": "Ratio paiement/facture acomptes",
    "cc_feat_0": "Ancienneté Carte de Crédit",
    "cc_feat_1": "Nb de transactions Carte de Crédit",
    "cc_feat_2": "Montant solde Carte de Crédit",
    "cc_feat_3": "Utilisation de la limite Carte de Crédit",
    "cc_feat_4": "Paiements en retard Carte de Crédit",
    
    # Mapping pour les variables catégorielles
    "NAME_CONTRACT_TYPE": "Type de Contrat",
    "CODE_GENDER": "Sexe",
    "FLAG_OWN_CAR": "Possède une Voiture",
    "NAME_INCOME_TYPE": "Type de Revenu",
}

# --- Fonctions Stub pour la Génération de Données (utilisées pour la démo) ---
@st.cache_data(show_spinner="Chargement des données...")
def load_application_data_stub(num_rows=None):
    """
    Charge et retourne les données depuis S3.
    Cette fonction ne s'exécutera qu'une seule fois.
    Args:
        num_rows (int, optional): Nombre de lignes à lire. Si None, lit 100 lignes.
    """
    try:
        # Définir le chemin local pour le dataset téléchargé
        dataset_local_path = os.path.join(LOCAL_DOWNLOAD_DIR, os.path.basename(DATASET_S3_KEY))
        
        # Télécharger le dataset depuis S3
        downloaded_path = download_file_from_s3(DATASET_S3_KEY, dataset_local_path)
        
        if downloaded_path is None:
            st.error("Impossible de télécharger le fichier de données depuis S3.")
            return None

        # Load data from the downloaded local path
        if num_rows is None:
            df = pd.read_csv(downloaded_path, nrows=100)
        else:
            df = pd.read_csv(downloaded_path, nrows=num_rows)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données depuis '{DATASET_S3_KEY}' : {e}")
        logger.exception("Erreur lors du chargement des données d'application:")
        return None

@st.cache_data(show_spinner="Traitement des données Bureau...")
def process_bureau_data_stub(df):
    if df is None: return None # Propagate None if previous step failed
    for i in range(5):
        if f'bureau_feat_{i}' not in df.columns:
            df[f'bureau_feat_{i}'] = np.random.rand(len(df))
    return df

@st.cache_data(show_spinner="Traitement des demandes précédentes...")
def process_previous_applications_data_stub(df):
    if df is None: return None # Propagate None if previous step failed
    for i in range(5):
        if f'prev_app_feat_{i}' not in df.columns:
            df[f'prev_app_feat_{i}'] = np.random.rand(len(df))
    return df

@st.cache_data(show_spinner="Traitement des données POS Cash...")
def process_pos_cash_data_stub(df):
    if df is None: return None # Propagate None if previous step failed
    for i in range(5):
        if f'pos_feat_{i}' not in df.columns:
            df[f'pos_feat_{i}'] = np.random.rand(len(df))
    return df

@st.cache_data(show_spinner="Traitement des paiements d'acomptes...")
def process_installments_payments_data_stub(df):
    if df is None: return None # Propagate None if previous step failed
    for i in range(5):
        if f'install_feat_{i}' not in df.columns:
            df[f'install_feat_{i}'] = np.random.rand(len(df))
    return df

@st.cache_data(show_spinner="Traitement des données de carte de crédit...")
def process_credit_card_balance_data_stub(df):
    if df is None: return None # Propagate None if previous step failed
    for i in range(5):
        if f'cc_feat_{i}' not in df.columns:
            df[f'cc_feat_{i}'] = np.random.rand(len(df))
    return df

@st.cache_data(show_spinner="Exécution du pipeline d'ingénierie des caractéristiques...")
def run_feature_engineering_pipeline(num_rows):
    # Pass num_rows to load_application_data_stub to limit initial data load
    df = load_application_data_stub(num_rows=num_rows)
    if df is None:
        return None # Return None if initial data loading failed
    df = process_bureau_data_stub(df)
    if df is None: return None
    df = process_previous_applications_data_stub(df)
    if df is None: return None
    df = process_pos_cash_data_stub(df)
    if df is None: return None
    df = process_installments_payments_data_stub(df)
    if df is None: return None
    df = process_credit_card_balance_data_stub(df)
    return df

# --- 5. Loading Functions (Cached) ---
# Cette fonction est modifiée pour charger les métadonnées en dur
# car nous ne nous connectons plus à un serveur MLflow distant.
@st.cache_resource(show_spinner="Chargement des métadonnées du modèle...")
def load_model_metadata_local():
    # Utilisation des dictionnaires définis directement dans le script
    features_info = SHAP_IMPORTANT_FEATURES_INFO
    optimal_threshold = 0.5 # Valeur par défaut, ajustez si nécessaire
    
    # Génération des noms de colonnes via le stub pour simuler les features d'entraînement
    # Only need 1 row to get column names, so this is efficient.
    dummy_data = run_feature_engineering_pipeline(num_rows=1)
    if dummy_data is None:
        logger.error("Failed to load dummy data for model metadata. Returning None for metadata.")
        return None, None, None # Return None for all values to indicate failure

    all_training_features = list(dummy_data.columns)
    
    logger.info("Métadonnées du modèle chargées localement.")
    return features_info, optimal_threshold, all_training_features

@st.cache_resource(show_spinner="Chargement du pipeline du modèle...")
def load_mlflow_pipeline_local():
    """
    Charge le pipeline MLflow en le téléchargeant depuis S3.
    """
    # Define the local path where the MLflow model directory will be downloaded
    model_local_dir = os.path.join(LOCAL_DOWNLOAD_DIR, os.path.basename(MODEL_S3_KEY_PREFIX.strip('/')))

    # Download the entire MLflow model directory from S3
    if not download_s3_directory(MODEL_S3_KEY_PREFIX, model_local_dir):
        st.error("Impossible de télécharger le dossier du modèle MLflow depuis S3.")
        return None

    try:
        # Now load the model from the local downloaded directory
        pipeline = mlflow.pyfunc.load_model(model_uri=model_local_dir)
        logger.info(f"Streamlit: Pipeline chargé depuis '{model_local_dir}'.")
        return pipeline
    except Exception as e:
        st.error(f"Échec lors du chargement du pipeline MLflow localement après téléchargement S3: {e}")
        st.info(f"Assurez-vous que le dossier '{model_local_dir}' existe et contient un modèle MLflow valide.")
        logger.exception("Erreur lors du chargement du pipeline MLflow:")
        return None

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
    # Tente d'extraire le pipeline scikit-learn sous-jacent
    # Si le modèle a été enregistré avec mlflow.sklearn.log_model, _model_impl devrait être le pipeline sklearn
    sklearn_pipeline = None
    if hasattr(_pyfunc_pipeline, '_model_impl'):
        sklearn_pipeline = _pyfunc_pipeline._model_impl
    
    if sklearn_pipeline is None or not hasattr(sklearn_pipeline, 'named_steps'):
        st.error("Impossible d'extraire le pipeline scikit-learn du modèle MLflow. L'explainer SHAP ne peut pas être initialisé correctement.")
        logger.error("Could not extract a scikit-learn Pipeline with named_steps from the MLflow PyFuncModel.")
        return None, None # Retourne None pour l'explainer et le préprocesseur en cas d'échec

    # Maintenant, utilise sklearn_pipeline pour obtenir le préprocesseur et le modèle final
    if 'preprocessor' in sklearn_pipeline.named_steps:
        preprocessor = sklearn_pipeline.named_steps['preprocessor']
    else:
        logger.warning("Le pipeline scikit-learn ne contient pas d'étape nommée 'preprocessor'. SHAP pourrait nécessiter un ajustement.")
        # Fallback vers un IdentityPreprocessor si l'étape 'preprocessor' n'est pas trouvée
        class IdentityPreprocessor:
            def transform(self, X): return X
            def get_feature_names_out(self, input_features=None):
                if input_features is not None and isinstance(input_features, list):
                    return input_features
                return [] # Fallback, can't infer from X here
        preprocessor = IdentityPreprocessor()
    
    final_model = sklearn_pipeline.steps[-1][1] # La dernière étape du pipeline sklearn est le modèle final

    # Génère les données de référence pour l'explainer SHAP
    ref_data_raw = run_feature_engineering_pipeline(num_rows=1000) # Garde 1000 lignes pour la référence SHAP
    if ref_data_raw is None:
        st.error("Impossible de charger les données de référence pour l'explainer SHAP.")
        return None, None

    ref_data_raw_filtered = ref_data_raw[all_training_features]
    ref_data_processed = preprocessor.transform(ref_data_raw_filtered)
    
    try:
        if hasattr(preprocessor, 'get_feature_names_out') and callable(preprocessor.get_feature_names_out):
            try:
                processed_feature_names = preprocessor.get_feature_names_out(input_features=all_training_features)
            except TypeError:
                processed_feature_names = preprocessor.get_feature_names_out()
        else:
            processed_feature_names = [f"col_{i}" for i in range(ref_data_processed.shape[1])]
            logger.warning("Impossible d'obtenir les noms de features du préprocesseur. Noms génériques utilisés.")
    except Exception as e:
        logger.warning(f"Erreur lors de l'appel de get_feature_names_out: {e}. Noms génériques utilisés.")
        processed_feature_names = [f"col_{i}" for i in range(ref_data_processed.shape[1])]

    ref_data_df = pd.DataFrame(ref_data_processed, columns=processed_feature_names)
    
    explainer = shap.Explainer(final_model, ref_data_df)
    return explainer, preprocessor

@st.cache_data(show_spinner="Génération des données de référence pour le drift...")
def load_reference_data_for_drift():
    try:
        # Utilise la fonction stub pour générer des données de référence
        # RÉDUIT LE NOMBRE DE LIGNES POUR ÉCONOMISER LA MÉMOIRE SUR STREAMLIT CLOUD
        reference_df = run_feature_engineering_pipeline(num_rows=5000) # Réduit de 30000 à 5000
        if reference_df is None:
            st.error("Impossible de générer le rapport Evidently : données de référence manquantes.")
            return None
        logger.info(f"Données de référence chargées avec succès. Nombre d'échantillons: {len(reference_df)}")
        return reference_df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données de référence : {e}")
        logger.exception("Erreur lors du chargement des données de référence dans Streamlit:")
        return None

# --- Fonctions d'Affichage des Rapports ---
def generate_and_display_evidently_report(reference_df, current_df):
    try:
        if reference_df is None or current_df is None:
            st.warning("Impossible de générer le rapport Evidently : données de référence ou actuelles manquantes.")
            return

        st.info("Génération du rapport en cours. Cela peut prendre quelques instants...")
        report_file_path = os.path.join(LOCAL_DOWNLOAD_DIR, "evidently_data_drift_report_temp.html")
        data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
        data_drift_dashboard.calculate(reference_data=reference_df, current_data=current_df)
        data_drift_dashboard.save(report_file_path)
        with open(report_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=1000, scrolling=True)
        st.success("Rapport Evidently généré et affiché avec succès.")
        # os.remove(report_file_path) # Optionally remove the file after display
    except Exception as e:
        st.error(f"Erreur lors de la génération ou de l'affichage du rapport Evidently : {e}")
        logger.exception("Erreur lors de l'exécution du rapport Evidently dans Streamlit:")

def map_feature_names(processed_feature_names, name_mapping):
    """
    Remplace les noms de colonnes transformés par des noms lisibles en utilisant
    un dictionnaire de mapping plus complet.
    """
    readable_names = []
    for name in processed_feature_names:
        # Nettoyer les préfixes du préprocesseur
        base_name = name.split('__')[-1]
        
        # 1. Chercher une correspondance exacte dans le dictionnaire complet
        if base_name in name_mapping:
            readable_name = name_mapping[base_name]
        # 2. Chercher une correspondance pour les variables catégorielles (ex: NAME_CONTRACT_TYPE_Cash)
        elif '_' in base_name:
            parts = base_name.split('_')
            original_name = '_'.join(parts[:-1]) # Variable d'origine (ex: NAME_CONTRACT_TYPE)
            category = parts[-1] # Catégorie (ex: Cash)
            if original_name in name_mapping:
                readable_name = f"{name_mapping[original_name]} : {category}"
            else:
                readable_name = base_name
        # 3. Utiliser le nom brut s'il n'y a pas de correspondance
        else:
            readable_name = base_name
            
        readable_names.append(readable_name)
    return readable_names

# Fonction utilitaire pour afficher les plots SHAP
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def display_shap_plot(shap_explainer, input_df, all_training_features, preprocessor):
    """Génère et affiche le force plot SHAP pour une seule prédiction."""
    st.subheader("📊 Explication de la Prédiction (SHAP)")
    st.info("""
        Le graphique SHAP ci-dessous montre comment chaque caractéristique a contribué à la prédiction.
        -   **Les valeurs rouges** poussent la prédiction vers un risque élevé.
        -   **Les valeurs bleues** poussent la prédiction vers un risque faible.
        -   **f(x)** est la probabilité prédite par le modèle.
        -   **E[f(x)]** est la probabilité moyenne du modèle sur l'ensemble de l'entraînement.
    """)
    try:
        if shap_explainer is None:
            st.error("L'explainer SHAP n'a pas pu être chargé. Impossible d'afficher le graphique SHAP.")
            return

        # Assurez-vous que input_df contient toutes les colonnes attendues par le préprocesseur
        # en utilisant les valeurs par défaut (0.0) pour les features non saisies
        full_input_df = pd.DataFrame(columns=all_training_features)
        full_input_df.loc[0] = 0 # Initialise avec des zéros
        for col in input_df.columns:
            if col in full_input_df.columns:
                full_input_df[col] = input_df[col]

        input_for_shap = preprocessor.transform(full_input_df[all_training_features])
        shap_values_instance = shap_explainer(input_for_shap)
        
        # Ensure processed_feature_names is obtained correctly from the preprocessor
        try:
            if hasattr(preprocessor, 'get_feature_names_out') and callable(preprocessor.get_feature_names_out):
                try:
                    processed_feature_names = preprocessor.get_feature_names_out(input_features=all_training_features)
                except TypeError:
                    processed_feature_names = preprocessor.get_feature_names_out()
            else:
                processed_feature_names = [f"col_{i}" for i in range(input_for_shap.shape[1])]
                logger.warning("Impossible d'obtenir les noms de features du préprocesseur pour SHAP. Noms génériques utilisés.")
        except Exception as e:
            logger.warning(f"Erreur lors de l'appel de get_feature_names_out pour SHAP: {e}. Noms génériques utilisés.")
            processed_feature_names = [f"col_{i}" for i in range(input_for_shap.shape[1])]

        readable_feature_names = map_feature_names(processed_feature_names, FULL_DESCRIPTIVE_NAMES)
        
        # Assurez-vous que les noms de features sont correctement mappés pour l'affichage SHAP
        processed_features_series = pd.Series(shap_values_instance.data[0], index=readable_feature_names)
        
        fig = shap.force_plot(
            base_value=shap_explainer.expected_value[0] if isinstance(shap_explainer.expected_value, np.ndarray) else shap_explainer.expected_value,
            shap_values=shap_values_instance.values[0],
            features=processed_features_series,
            matplotlib=False
        )
        st_shap(fig, height=250)
    except Exception as e:
        st.error(f"Erreur lors de la génération du graphique SHAP : {e}")
        logger.exception("Erreur lors de l'exécution de SHAP dans Streamlit:")

# --- 6. Chargement des Ressources au Démarrage ---
# Ces fonctions vont maintenant télécharger les données/modèles depuis S3
features_info, optimal_threshold, all_training_features = load_model_metadata_local()
pipeline = load_mlflow_pipeline_local()

# --- 7. Contenu Principal de la Page Streamlit ---
st.title("📊 Prédiction de Défaut Client & Surveillance du Modèle")

tab1, tab2 = st.tabs(["Prédiction de Prêt", "Analyse du Data Drift"])

with tab1:
    st.markdown("""
    Cette application vous permet de simuler une prédiction de risque de défaut pour un client.
    """)
    # Vérifie que le modèle et les métadonnées sont chargés
    if features_info is None or pipeline is None:
        st.error("L'application n'a pas pu être initialisée car le modèle ou les métadonnées n'ont pas pu être chargés. Veuillez vérifier les logs pour plus de détails.")
        st.stop() # Arrête l'exécution si les ressources critiques sont manquantes
    else:
        st.sidebar.header("Informations sur le Modèle")
        st.sidebar.write(f"**Nom du Modèle :** `Pipeline de Régression Logistique`") # Nom générique
        st.sidebar.write(f"**Source du Modèle :** `AWS S3 (Bucket: {S3_BUCKET_NAME})`") # Updated source
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
            # Créer un DataFrame avec toutes les colonnes attendues par le modèle
            # en utilisant les valeurs par défaut (0.0) pour les features non saisies
            model_input_data = {
                feature_name: user_inputs.get(feature_name, 0.0)
                for feature_name in all_training_features
            }
            input_df = pd.DataFrame([model_input_data])
            
            try:
                # Charger l'explainer SHAP et le préprocesseur
                # Ces fonctions sont maintenant appelées ici, après avoir vérifié que pipeline est non-None
                shap_explainer, preprocessor_for_shap = load_shap_explainer(pipeline, all_training_features)
                
                if shap_explainer is None or preprocessor_for_shap is None:
                    st.error("Impossible d'initialiser l'explication SHAP. Veuillez vérifier les logs.")
                else:
                    prediction_proba = pipeline.predict_proba(input_df)[:, 1][0]
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
                    
                    display_shap_plot(shap_explainer, input_df, all_training_features, preprocessor_for_shap)
                
            except Exception as e:
                st.error(f"Une erreur est survenue lors de l'exécution de la prédiction : {e}")
                logger.exception("Erreur lors de la prédiction Streamlit:")

with tab2:
    st.header("Analyse du Data Drift (Evidently AI)")
    st.markdown("""
    Cette section génère et affiche un rapport de **Data Drift** directement dans l'application.
    Le rapport compare les données d'entraînement (référence) aux données de production simulées.
    """)
    
    if st.button("Générer et afficher le rapport de Data Drift"):
        reference_data_for_drift = load_reference_data_for_drift()
        
        # Simulation de data drift
        if reference_data_for_drift is not None:
            df_production = reference_data_for_drift.copy()
            if 'AMT_CREDIT' in df_production.columns:
                df_production['AMT_CREDIT'] = df_production['AMT_CREDIT'] * np.random.normal(1.2, 0.1, len(df_production))
            if 'app_feature_17' in df_production.columns: # Correspond à l'âge client
                df_production['app_feature_17'] = df_production['app_feature_17'] + np.random.randint(-5, 5, len(df_production))
            
            generate_and_display_evidently_report(reference_data_for_drift, df_production)
        else:
            st.warning("Impossible de générer le rapport de Data Drift car les données de référence n'ont pas pu être chargées.")
    else:
        st.warning("Cliquez sur le bouton pour générer le rapport de dérive des données.")
