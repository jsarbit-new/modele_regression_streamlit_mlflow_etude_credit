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
    aws_secret_access_key=AWS_SECRET_ACCESS_key
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
        metadata = pd.read_json(io.BytesIO(obj['Body'].read())) # Utilisez pd.read_json pour la simplicité
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
    if not os.path.exists(model_local_dir):
        # Tenter de télécharger si le répertoire n'existe pas localement
        st.warning(f"Répertoire du modèle MLflow '{model_local_dir}' non trouvé localement. Tentative de téléchargement depuis S3...")
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
    if os.path.exists(local_dir):
        logger.info(f"Le répertoire local '{local_dir}' existe déjà, pas de téléchargement nécessaire.")
        return

    os.makedirs(local_dir, exist_ok=True)
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=AWS_S3_BUCKET_NAME, Prefix=s3_prefix)

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
            else:
                st.warning(f"Aucun objet trouvé sous le préfixe S3: {s3_prefix}")
                logger.warning(f"No objects found under S3 prefix: {s3_prefix}")
                return # Sortir si rien n'est trouvé
        logger.info(f"Modèle MLflow téléchargé avec succès depuis S3 '{s3_prefix}' vers '{local_dir}'.")
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

@st.cache_data(show_spinner="Chargement des données d'entraînement pour SHAP et Data Drift...")
def load_training_data_for_shap(file_key="data/application_train.csv"):
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

# Note : Dans une application réelle, cette fonction devrait répliquer EXACTEMENT
# l'étape de pré-traitement de votre pipeline scikit-learn avant le modèle.
# Ici, nous allons la laisser simple pour correspondre aux colonnes brutes.
# Si votre pipeline inclut un préprocesseur qui modifie le nombre/nom des colonnes,
# cette fonction devrait le refléter pour que SHAP soit précis sur les features pré-traitées.

@st.cache_data(show_spinner="Préparation des données pour l'ingénierie des caractéristiques...")
def run_feature_engineering_pipeline(num_rows=None):
    """
    Simule une partie de l'ingénierie des caractéristiques pour obtenir des données brutes
    à utiliser comme base pour SHAP et le préprocesseur.
    """
    raw_data = load_training_data_for_shap()
    if raw_data is None:
        return None
    
    # Ici, vous devriez inclure les étapes de votre pipeline qui transforment les données
    # brutes en données que votre modèle attend.
    # Pour l'instant, on retourne un échantillon des données brutes.
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
    # Tente d'extraire le modèle scikit-learn sous-jacent
    # Si le modèle a été enregistré avec mlflow.sklearn.log_model, _model_impl devrait être le modèle/pipeline sklearn
    sklearn_model_or_pipeline = None
    if hasattr(_pyfunc_pipeline, '_model_impl'):
        sklearn_model_or_pipeline = _pyfunc_pipeline._model_impl
    
    if sklearn_model_or_pipeline is None:
        st.error("Impossible d'extraire le modèle/pipeline scikit-learn du modèle MLflow. L'explainer SHAP ne peut pas être initialisé correctement.")
        logger.error("Could not extract a scikit-learn model or Pipeline from the MLflow PyFuncModel.")
        return None, None # Retourne None pour l'explainer et le préprocesseur en cas d'échec

    preprocessor = None
    final_model = None

    # Vérifie si c'est un Pipeline scikit-learn
    if isinstance(sklearn_model_or_pipeline, Pipeline) or hasattr(sklearn_model_or_pipeline, 'named_steps'):
        logger.info("Modèle MLflow chargé est reconnu comme un Pipeline scikit-learn.")
        # C'est un pipeline, extrayons le préprocesseur et le modèle final
        if 'preprocessor' in sklearn_model_or_pipeline.named_steps:
            preprocessor = sklearn_model_or_pipeline.named_steps['preprocessor']
            logger.info("Préprocesseur nommé 'preprocessor' trouvé dans le pipeline.")
        else:
            logger.warning("Le pipeline scikit-learn ne contient pas d'étape nommée 'preprocessor'. Utilisation d'IdentityPreprocessor.")
            preprocessor = IdentityPreprocessor()
        
        # Le modèle final est généralement la dernière étape du pipeline
        final_model = sklearn_model_or_pipeline.steps[-1][1] 
        logger.info(f"Modèle final extrait du pipeline: {type(final_model)}")
    else:
        # Si ce n'est pas un pipeline avec named_steps, c'est probablement le modèle final directement
        logger.info("Le modèle MLflow chargé n'est pas un pipeline scikit-learn avec 'named_steps'. Traitement comme un modèle final direct.")
        final_model = sklearn_model_or_pipeline
        # Dans ce cas, nous n'avons pas de préprocesseur explicite, utilisons un IdentityPreprocessor
        preprocessor = IdentityPreprocessor()

    if final_model is None:
        st.error("Impossible d'identifier le modèle final pour l'explainer SHAP.")
        logger.error("Final model could not be extracted or identified.")
        return None, None

    # Génère les données de référence pour l'explainer SHAP
    ref_data_raw = run_feature_engineering_pipeline(num_rows=1000) # Garde 1000 lignes pour la référence SHAP
    if ref_data_raw is None:
        st.error("Impossible de charger les données de référence pour l'explainer SHAP.")
        return None, None

    # S'assurer que seules les colonnes d'entraînement sont utilisées
    ref_data_raw_filtered = ref_data_raw[all_training_features]
    
    # Appliquer le préprocesseur pour obtenir les données traitées
    ref_data_processed = preprocessor.transform(ref_data_raw_filtered)
    
    # Obtenir les noms des caractéristiques après prétraitement
    processed_feature_names = all_training_features # Fallback
    try:
        if hasattr(preprocessor, 'get_feature_names_out') and callable(preprocessor.get_feature_names_out):
            # Tente avec input_features pour ColumnTransformer
            try:
                processed_feature_names = preprocessor.get_feature_names_out(input_features=all_training_features)
            except TypeError: # Si get_feature_names_out ne prend pas input_features
                processed_feature_names = preprocessor.get_feature_names_out()
        elif hasattr(ref_data_processed, 'columns'): # Si c'est un DataFrame après transformation
            processed_feature_names = ref_data_processed.columns.tolist()
        else:
            processed_feature_names = [f"col_{i}" for i in range(ref_data_processed.shape[1])]
            logger.warning("Impossible d'obtenir les noms de features du préprocesseur. Noms génériques utilisés.")
    except Exception as e:
        logger.warning(f"Erreur lors de l'appel de get_feature_names_out ou extraction des colonnes: {e}. Noms génériques utilisés.")
        processed_feature_names = [f"col_{i}" for i in range(ref_data_processed.shape[1])]


    # Assurez-vous que ref_data_processed est un DataFrame pour SHAP
    ref_data_df = pd.DataFrame(ref_data_processed, columns=processed_feature_names)
    
    # Utilisation de shap.Explainer avec le modèle final et les données de référence
    explainer = shap.Explainer(final_model, ref_data_df)
    return explainer, preprocessor


# --- Fonctions d'affichage ---

def plot_feature_importance(explainer, shap_values, feature_names, top_n=10):
    """Affiche les importances globales des caractéristiques SHAP."""
    if explainer is None or shap_values is None or not feature_names:
        st.warning("Impossible d'afficher l'importance des caractéristiques : données SHAP manquantes.")
        return

    # S'assurer que shap_values a la bonne forme et que c'est bien numpy array
    if isinstance(shap_values, list): # Si c'est une liste de valeurs SHAP (pour plusieurs sorties)
        shap_values_abs_mean = np.abs(np.array(shap_values[0])).mean(0)
    else: # Pour une seule sortie ou direct
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
    
    # Assurez-vous que shap_values_individual est bien un tableau numpy
    if isinstance(shap_values_individual, list):
        # Pour le cas où le modèle a plusieurs sorties, prenons la première pour l'explication individuelle
        shap_values_individual = shap_values_individual[0] 

    try:
        # Utilisez shap.waterfall_plot directement.
        # Il nécessite un shap.Explanation objet, qui peut être créé.
        # Si explainer est un TreeExplainer ou KernelExplainer, base_values est direct.
        # Sinon, il faut le récupérer depuis l'explainer ou le calculer (e.g., np.mean(explainer.expected_value)).
        
        # Pour shap.Explainer générique, expected_value est un attribut
        expected_value = explainer.expected_value
        
        # S'assurer que processed_features_df est un DataFrame pandas pour l'indexation
        if not isinstance(processed_features_df, pd.DataFrame):
            processed_features_df = pd.DataFrame([processed_features_df], columns=feature_names)
            
        shap.initjs() # Initialise JavaScript pour les plots SHAP
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_individual, 
                base_values=expected_value, 
                data=processed_features_df.iloc[0].values, 
                feature_names=feature_names
            ),
            max_display=20,
            show=False, # Important: ne pas afficher directement, laisser Streamlit le faire
            ax=ax
        )
        ax.set_title(f"Explication SHAP pour le client {client_id}")
        st.pyplot(fig) # Utilisez st.pyplot pour afficher le plot matplotlib
        plt.close(fig) # Fermez la figure pour libérer la mémoire

    except Exception as e:
        st.error(f"Erreur lors de l'affichage de l'explication SHAP individuelle : {e}")
        logger.error(f"Error plotting individual SHAP explanation: {e}", exc_info=True)


# --- Fonctions principales de l'application Streamlit ---

st.set_page_config(layout="wide", page_title="Prédiction de Risque de Crédit et Explicabilité")

st.title("Tableau de Bord de Prédiction de Risque de Crédit")
st.write("Cette application prédit le risque de défaut de paiement pour les demandes de crédit et fournit des explications sur les prédictions.")

# Chargement des données et du modèle
data = load_data_from_s3("data/application_test.csv")
if data is None:
    st.stop() # Arrêter l'exécution si les données ne sont pas chargées

# Chargement des métadonnées du modèle
model_metadata = load_model_metadata_from_s3()
if model_metadata is None:
    st.error("Impossible de charger les métadonnées du modèle. Certaines fonctionnalités pourraient être limitées.")
    # Définir des valeurs par défaut si les métadonnées ne sont pas disponibles
    all_training_features = data.columns.tolist() # Fallback, à ajuster si besoin
    threshold = 0.5 # Valeur par défaut si non trouvée
else:
    all_training_features = model_metadata['training_features'].tolist() if 'training_features' in model_metadata else data.columns.tolist()
    threshold = model_metadata.get('threshold', 0.5) # Récupère le seuil ou utilise 0.5 par défaut
    
    # Assurez-vous que les colonnes nécessaires sont présentes dans 'data'
    missing_features_in_data = [f for f in all_training_features if f not in data.columns]
    if missing_features_in_data:
        st.warning(f"Attention : Les caractéristiques suivantes du modèle sont manquantes dans les données de test : {', '.join(missing_features_in_data)}. Le modèle pourrait ne pas fonctionner comme prévu.")
        # Filtrer all_training_features pour ne garder que celles qui sont dans `data`
        all_training_features = [f for f in all_training_features if f in data.columns]


# Chargement du pipeline MLflow
pipeline = load_mlflow_pipeline_local()
if pipeline is None:
    st.stop() # Arrêter l'exécution si le pipeline ne peut pas être chargé


# --- Sidebar pour la sélection du client ---
st.sidebar.header("Sélection du Client")
client_ids = data['SK_ID_CURR'].tolist()
selected_client_id = st.sidebar.selectbox("Sélectionnez un ID Client :", client_ids)

# Trouver les données du client sélectionné
client_data_raw = data[data['SK_ID_CURR'] == selected_client_id].drop(columns=['SK_ID_CURR']).iloc[0]
client_data_df_input = pd.DataFrame([client_data_raw]) # Pour l'entrée du pipeline

# Prétraiter les données du client pour la prédiction
# Assurez-vous que seules les features attendues par le pipeline sont passées
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
            # SHAP pour les modèles binaires de classification donne souvent deux tableaux de valeurs (pour la classe 0 et la classe 1)
            # Nous nous intéressons à la classe positive (par défaut, classe 1)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                individual_shap_values = shap_values[1][0] # Pour la classe 1, et le premier échantillon
            else:
                individual_shap_values = shap_values[0] # Si c'est un tableau unique (régression ou 1D classification)
            
            # Convertir en DataFrame pour l'affichage si nécessaire
            client_processed_df = pd.DataFrame(client_data_processed_for_shap, columns=processed_feature_names)

            plot_individual_explanation(explainer, individual_shap_values, client_processed_df, processed_feature_names, selected_client_id)
            
            # Affichage de l'importance globale des features (si plusieurs clients sont prévus pour cela, sinon ce serait un plot sur les données d'entraînement)
            # Pour l'instant, on peut réutiliser les shap values calculées pour le client unique, mais idéalement cela devrait venir d'un ensemble de données.
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