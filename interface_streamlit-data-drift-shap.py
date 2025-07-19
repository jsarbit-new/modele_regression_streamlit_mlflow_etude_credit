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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder # OneHotEncoder si votre FE l'utilise
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import json
import gc # Pour la gestion de la mémoire

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

# --- Fonctions de chargement de données depuis S3 ---

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
            st.error(f"Le fichier '{file_key}' est introuvable dans le bucket S3. Veuillez vérifier le chemin ou le nom du fichier.")
            logger.error(f"S3 file not found: {file_key}")
        else:
            st.error(f"Erreur lors du chargement depuis S3: {e}")
            logger.error(f"Error loading from S3: {e}")
        return None
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue lors du chargement des données depuis S3: {e}")
        logger.error(f"Unexpected error loading data from S3: {e}")
        return None

# --- Chargement local des ressources MLflow ---

@st.cache_resource(show_spinner="Chargement du pipeline MLflow localement...")
def load_mlflow_pipeline_local(model_local_dir="./downloaded_assets/modele_mlflow"):
    """
    Charge un pipeline MLflow depuis un répertoire local et extrait ses métadonnées et sa signature.
    Args:
        model_local_dir (str): Chemin local du répertoire du modèle MLflow.
    Returns:
        tuple: (pipeline MLflow, dictionnaire des métadonnées du modèle, liste des noms de features d'entrée) ou (None, None, None) en cas d'échec.
    """
    if not os.path.exists(model_local_dir) or not os.listdir(model_local_dir):
        st.warning(f"Répertoire du modèle MLflow '{model_local_dir}' non trouvé ou vide localement. Tentative de téléchargement depuis S3...")
        try:
            download_mlflow_model_from_s3("modele_mlflow", model_local_dir)
        except Exception as e:
            st.error(f"Échec du téléchargement du modèle MLflow depuis S3 : {e}")
            return None, None, None

    try:
        pipeline = mlflow.pyfunc.load_model(model_uri=model_local_dir)
        
        model_metadata_from_pipeline = {}
        if hasattr(pipeline, 'metadata') and pipeline.metadata:
            for key, value in pipeline.metadata.to_dict().items():
                try:
                    model_metadata_from_pipeline[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    model_metadata_from_pipeline[key] = value
        
        # Tenter d'extraire les noms de features de input_example.json (plus fiable)
        input_features_from_example = []
        input_example_path = os.path.join(model_local_dir, "input_example.json")
        if os.path.exists(input_example_path):
            try:
                with open(input_example_path, 'r') as f:
                    input_example_data = json.load(f)
                # Assuming input_example.json contains a dict or list of dicts
                if isinstance(input_example_data, dict):
                    input_features_from_example = list(input_example_data.keys())
                elif isinstance(input_example_data, list) and input_example_data:
                    # If it's a list of examples, take keys from the first one
                    input_features_from_example = list(input_example_data[0].keys())
                logger.info(f"Features d'entrée récupérées de input_example.json: {input_features_from_example}")
            except Exception as e:
                logger.warning(f"Erreur lors de la lecture ou du parsing de input_example.json: {e}")
        else:
            logger.warning("Fichier input_example.json non trouvé dans le répertoire du modèle MLflow.")

        # Tenter d'extraire les noms de features de la signature du modèle (fallback)
        input_features_from_signature = []
        if not input_features_from_example and hasattr(pipeline, 'signature') and pipeline.signature and pipeline.signature.inputs:
            input_features_from_signature = pipeline.signature.inputs.column_names()
            logger.info(f"Features d'entrée récupérées de la signature MLflow: {input_features_from_signature}")
        elif not input_features_from_example:
            logger.warning("Impossible de récupérer les features d'entrée de la signature MLflow.")

        # Prioriser input_features_from_example, puis signature
        final_input_features = input_features_from_example or input_features_from_signature or []

        logger.info("Streamlit: Pipeline chargé depuis './downloaded_assets/modele_mlflow'.")
        return pipeline, model_metadata_from_pipeline, final_input_features
    except Exception as e:
        logger.error(f"Erreur lors du chargement du pipeline MLflow: {e}", exc_info=True)
        st.error(f"Erreur lors du chargement du pipeline MLflow : {e}. Assurez-vous que le modèle est compatible avec les dépendances installées.")
        return None, None, None

@st.cache_resource(show_spinner="Téléchargement du modèle MLflow depuis S3...")
def download_mlflow_model_from_s3(s3_prefix, local_dir):
    """Télécharge un modèle MLflow complet (répertoire) depuis S3."""
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
                    local_file_path = os.path.join(local_dir, os.path.relpath(object_key, s3_prefix))
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                    if not object_key.endswith('/'):
                        s3_client.download_file(AWS_S3_BUCKET_NAME, object_key, local_file_path)
                        files_downloaded_count += 1
            else:
                st.warning(f"Aucun objet trouvé sous le préfixe S3: {s3_prefix}")
                logger.warning(f"No objects found under S3 prefix: {s3_prefix}")
                return
        
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

# --- Fonctions de Feature Engineering (Réelles) ---

# Fonction one_hot_encoder (copiée de votre code)
def one_hot_encoder(df, nan_as_category = True):
    """
    Applique l'encodage one-hot aux colonnes de type 'object' d'un DataFrame.
    Gère également les NaN comme une catégorie si nan_as_category est True.
    """
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

@st.cache_data(show_spinner="Traitement des données d'application (FE)...")
def process_application_data_streamlit(df_app_raw):
    """
    TRAITEMENT DES DONNÉES APPLICATION_TRAIN/TEST.
    LOGIQUE COPIÉE DE VOTRE FONCTION application_train_test.
    """
    logger.info(f"Début du traitement des données application. Forme: {df_app_raw.shape}")
    
    # Votre logique de application_train_test
    df = df_app_raw.copy() # Travailler sur une copie pour ne pas modifier le DataFrame mis en cache

    # Gérer 'CODE_GENDER' == 'XNA'
    df = df[df['CODE_GENDER'] != 'XNA']

    # Factorize pour les features binaires
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    
    # One-hot encoding pour les features catégorielles
    df, cat_cols = one_hot_encoder(df, nan_as_category=False) # nan_as_category=False comme dans votre fonction

    # Remplacer 365243 par NaN pour 'DAYS_EMPLOYED'
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # Création de nouvelles features
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    
    # Gérer la division par zéro avant de créer les features
    df['INCOME_CREDIT_PERC'] = np.where(df['AMT_CREDIT'] == 0, np.nan, df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT'])
    df['INCOME_PER_PERSON'] = np.where(df['CNT_FAM_MEMBERS'] == 0, np.nan, df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'])
    df['ANNUITY_INCOME_PERC'] = np.where(df['AMT_INCOME_TOTAL'] == 0, np.nan, df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'])
    df['PAYMENT_RATE'] = np.where(df['AMT_CREDIT'] == 0, np.nan, df['AMT_ANNUITY'] / df['AMT_CREDIT'])

    # Gérer les valeurs infinies qui peuvent apparaître après division
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    logger.info(f"Fin du traitement des données application. Forme: {df.shape}")
    return df

@st.cache_data(show_spinner="Traitement des données bureau et bureau_balance (FE)...")
def process_bureau_data_streamlit():
    """
    TRAITEMENT DES DONNÉES BUREAU ET BUREAU_BALANCE.
    LOGIQUE COPIÉE DE VOTRE FONCTION bureau_and_balance.
    """
    logger.info("Chargement et traitement des données bureau et bureau_balance.")
    bureau = load_data_from_s3("input/bureau.csv")
    bb = load_data_from_s3("input/bureau_balance.csv")

    if bureau is None or bb is None:
        st.error("Impossible de charger les données bureau ou bureau_balance.")
        return pd.DataFrame({'SK_ID_CURR': []}) # Retourne un DataFrame vide avec SK_ID_CURR

    # --- VOTRE VRAIE LOGIQUE DE FE POUR BUREAU VA ICI ---
    # Logique copiée de votre fonction bureau_and_balance
    bb, bb_cat = one_hot_encoder(bb, nan_as_category=True)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category=True)

    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']

    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])

    new_bb_agg_cols = {}
    for col in bb_agg.columns:
        original_bb_cat_name = col.replace('_MEAN', '')
        if original_bb_cat_name in bb_cat:
            new_bb_agg_cols[col] = col + '_BB'
        elif any(s in col for s in [cat.replace('_MEAN', '') for cat in bb_cat]):
            if '_BB_MEAN' not in col:
                new_bb_agg_cols[col] = col.replace('_MEAN', '_BB_MEAN') if '_MEAN' in col else col + '_BB_MEAN'

    bb_agg = bb_agg.rename(columns=new_bb_agg_cols)

    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']

    for col in bureau.columns:
        if '_BB_MEAN' in col:
            cat_aggregations[col] = ['mean']

    final_aggregations = {**num_aggregations, **cat_aggregations}

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(final_aggregations)
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()

    logger.info(f"Fin du traitement des données bureau. Forme: {bureau_agg.shape}")
    return bureau_agg.reset_index()

@st.cache_data(show_spinner="Traitement des applications précédentes (FE)...")
def process_previous_applications_streamlit():
    """
    TRAITEMENT DES DONNÉES PREVIOUS_APPLICATION.
    LOGIQUE COPIÉE DE VOTRE FONCTION previous_applications.
    """
    logger.info("Chargement et traitement des données previous_application.")
    prev = load_data_from_s3("input/previous_application.csv")
    if prev is None:
        st.error("Impossible de charger les données previous_application.")
        return pd.DataFrame({'SK_ID_CURR': []})

    # --- VOTRE VRAIE LOGIQUE DE FE POUR PREVIOUS_APPLICATION VA ICI ---
    # Logique copiée de votre fonction previous_applications
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)

    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    prev['APP_CREDIT_PERC'] = np.where(prev['AMT_CREDIT'] == 0, np.nan, prev['AMT_APPLICATION'] / prev['AMT_CREDIT'])

    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()

    logger.info(f"Fin du traitement des données previous_application. Forme: {prev_agg.shape}")
    return prev_agg.reset_index()

@st.cache_data(show_spinner="Traitement des données POS_CASH_balance (FE)...")
def process_pos_cash_streamlit():
    """
    TRAITEMENT DES DONNÉES POS_CASH_BALANCE.
    LOGIQUE COPIÉE DE VOTRE FONCTION pos_cash.
    """
    logger.info("Chargement et traitement des données POS_CASH_balance.")
    pos = load_data_from_s3("input/POS_CASH_balance.csv")
    if pos is None:
        st.error("Impossible de charger les données POS_CASH_balance.")
        return pd.DataFrame({'SK_ID_CURR': []})

    # --- VOTRE VRAIE LOGIQUE DE FE POUR POS_CASH_BALANCE VA ICI ---
    # Logique copiée de votre fonction pos_cash
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)

    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()

    logger.info(f"Fin du traitement des données POS_CASH_balance. Forme: {pos_agg.shape}")
    return pos_agg.reset_index()

@st.cache_data(show_spinner="Traitement des données installments_payments (FE)...")
def process_installments_payments_streamlit():
    """
    TRAITEMENT DES DONNÉES INSTALLMENTS_PAYMENTS.
    LOGIQUE COPIÉE DE VOTRE FONCTION installments_payments.
    """
    logger.info("Chargement et traitement des données installments_payments.")
    ins = load_data_from_s3("input/installments_payments.csv")
    if ins is None:
        st.error("Impossible de charger les données installments_payments.")
        return pd.DataFrame({'SK_ID_CURR': []})

    # --- VOTRE VRAIE LOGIQUE DE FE POUR INSTALLMENTS_PAYMENTS VA ICI ---
    # Logique copiée de votre fonction installments_payments
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)

    ins['PAYMENT_PERC'] = np.where(ins['AMT_INSTALMENT'] == 0, np.nan, ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT'])
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()

    logger.info(f"Fin du traitement des données installments_payments. Forme: {ins_agg.shape}")
    return ins_agg.reset_index()

@st.cache_data(show_spinner="Traitement des données credit_card_balance (FE)...")
def process_credit_card_balance_streamlit():
    """
    TRAITEMENT DES DONNÉES CREDIT_CARD_BALANCE.
    LOGIQUE COPIÉE DE VOTRE FONCTION credit_card_balance.
    """
    logger.info("Chargement et traitement des données credit_card_balance.")
    cc = load_data_from_s3("input/credit_card_balance.csv")
    if cc is None:
        st.error("Impossible de charger les données credit_card_balance.")
        return pd.DataFrame({'SK_ID_CURR': []})

    # --- VOTRE VRAIE LOGIQUE DE FE POUR CREDIT_CARD_BALANCE VA ICI ---
    # Logique copiée de votre fonction credit_card_balance
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)

    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)

    num_aggregations = {
        'MONTHS_BALANCE': ['min', 'max', 'mean', 'sum', 'var'],
        'AMT_BALANCE': ['min', 'max', 'mean', 'sum', 'var'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_RECEIVABLE_PRINCIPAL': ['min', 'max', 'mean', 'sum'],
        'AMT_RECIVABLE': ['min', 'max', 'mean', 'sum'],
        'AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT_MATURE_CUM': ['min', 'max', 'mean', 'sum'],
        'SK_DPD': ['min', 'max', 'mean', 'sum'],
        'SK_DPD_DEF': ['min', 'max', 'mean', 'sum']
    }
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean', 'sum']

    cc_agg = cc.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()

    logger.info(f"Fin du traitement des données credit_card_balance. Forme: {cc_agg.shape}")
    return cc_agg.reset_index()

# --- Fonction pour obtenir le préprocesseur ajusté sur les données d'entraînement ---
@st.cache_resource(show_spinner="Préparation et ajustement du préprocesseur sur les données d'entraînement...")
def get_fitted_preprocessor():
    """
    Charge les données d'entraînement brutes, effectue le FE, et ajuste le ColumnTransformer.
    Ce préprocesseur ajusté sera réutilisé pour les données de test/prédiction.
    """
    logger.info("Début de l'ajustement du préprocesseur.")
    
    # 1. Chargement des données d'entraînement brutes
    df_train_raw = load_data_from_s3("input/application_train.csv")
    if df_train_raw is None:
        st.error("Impossible de charger les données d'entraînement brutes pour ajuster le préprocesseur.")
        return None

    # 2. Exécution du Feature Engineering pour les données d'entraînement
    # Récupérer les données auxiliaires transformées
    bureau_df_fe = process_bureau_data_streamlit()
    prev_app_df_fe = process_previous_applications_streamlit()
    pos_cash_df_fe = process_pos_cash_streamlit()
    install_payments_df_fe = process_installments_payments_streamlit()
    credit_card_df_fe = process_credit_card_balance_streamlit()

    # Jointure des DataFrames transformés
    df_fe = process_application_data_streamlit(df_train_raw.copy()) # Utilise une copie pour ne pas modifier l'original mis en cache
    
    # Jointures comme dans votre script MLflow
    df_fe = df_fe.set_index('SK_ID_CURR').join(bureau_df_fe.set_index('SK_ID_CURR'), how='left', rsuffix='_bureau').reset_index()
    df_fe = df_fe.set_index('SK_ID_CURR').join(prev_app_df_fe.set_index('SK_ID_CURR'), how='left', rsuffix='_prev').reset_index()
    df_fe = df_fe.set_index('SK_ID_CURR').join(pos_cash_df_fe.set_index('SK_ID_CURR'), how='left', rsuffix='_pos').reset_index()
    df_fe = df_fe.set_index('SK_ID_CURR').join(install_payments_df_fe.set_index('SK_ID_CURR'), how='left', rsuffix='_ins').reset_index()
    df_fe = df_fe.set_index('SK_ID_CURR').join(credit_card_df_fe.set_index('SK_ID_CURR'), how='left', rsuffix='_cc').reset_index()
    
    del bureau_df_fe, prev_app_df_fe, pos_cash_df_fe, install_payments_df_fe, credit_card_df_fe
    gc.collect()

    # Nettoyage des noms de colonnes (remplacement des caractères spéciaux pour compatibilité SHAP, etc.)
    clean_column_mapping = {col: "".join(c if c.isalnum() else "_" for c in str(col)) for col in df_fe.columns}
    df_fe.rename(columns=clean_column_mapping, inplace=True)
    
    # Convertir en numérique et gérer les infinis
    for col in df_fe.columns:
        df_fe[col] = pd.to_numeric(df_fe[col], errors='coerce')
    df_fe.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Exclure 'TARGET' et 'SK_ID_CURR' pour le prétraitement
    exclude_features_for_preprocessor = ['TARGET', 'SK_ID_CURR', 'index'] # 'index' peut parfois apparaître
    features_for_preprocessor = [f for f in df_fe.columns if f not in exclude_features_for_preprocessor]
    
    df_features_for_preprocessor = df_fe[features_for_preprocessor].copy()

    # Définition du pipeline de prétraitement pour les caractéristiques numériques
    numeric_features = df_features_for_preprocessor.columns.tolist() # Toutes les colonnes restantes sont numériques

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Impute les NaNs avec la médiane
        ('scaler', MinMaxScaler(feature_range=(0, 1))) # Scale les valeurs entre 0 et 1
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='passthrough' # Conserve les colonnes non spécifiées (aucune ici)
    )

    # Ajuste le préprocesseur sur les données d'entraînement
    preprocessor.fit(df_features_for_preprocessor)
    logger.info("Préprocesseur ajusté avec succès sur les données d'entraînement.")
    return preprocessor

# --- NOUVELLE FONCTION : Ingénierie des Caractéristiques dans Streamlit ---
@st.cache_data(show_spinner="Exécution de l'ingénierie des caractéristiques...")
def run_feature_engineering_streamlit(raw_df_app, fitted_preprocessor, expected_features, is_training_data=False):
    """
    Effectue l'ingénierie des caractéristiques sur les données brutes fournies.
    Cette fonction utilise le préprocesseur ajusté sur les données d'entraînement.

    Args:
        raw_df_app (pd.DataFrame): Le DataFrame de données brutes (application_train/test.csv).
        fitted_preprocessor (ColumnTransformer): Le préprocesseur ajusté sur les données d'entraînement.
        expected_features (list): Liste des noms de features attendus par le modèle MLflow.
        is_training_data (bool): Indique si les données sont des données d'entraînement (pour SHAP).
    Returns:
        pd.DataFrame: Le DataFrame avec les caractéristiques transformées et prétraitées.
    """
    logger.info(f"Début de l'ingénierie des caractéristiques dans Streamlit. Forme des données brutes app: {raw_df_app.shape}")

    # Assurez-vous que SK_ID_CURR est conservé pour la jointure ou l'identification
    sk_id_curr = None
    if 'SK_ID_CURR' in raw_df_app.columns:
        sk_id_curr = raw_df_app['SK_ID_CURR']
    
    # 1. Exécution du Feature Engineering pour les données d'application
    df_fe = process_application_data_streamlit(raw_df_app.copy())

    # 2. Récupérer et traiter les données auxiliaires
    bureau_df_fe = process_bureau_data_streamlit()
    prev_app_df_fe = process_previous_applications_streamlit()
    pos_cash_df_fe = process_pos_cash_streamlit()
    install_payments_df_fe = process_installments_payments_streamlit()
    credit_card_df_fe = process_credit_card_balance_streamlit()

    # 3. Jointure des DataFrames transformés (comme dans votre script MLflow)
    df_fe = df_fe.set_index('SK_ID_CURR').join(bureau_df_fe.set_index('SK_ID_CURR'), how='left', rsuffix='_bureau').reset_index()
    df_fe = df_fe.set_index('SK_ID_CURR').join(prev_app_df_fe.set_index('SK_ID_CURR'), how='left', rsuffix='_prev').reset_index()
    df_fe = df_fe.set_index('SK_ID_CURR').join(pos_cash_df_fe.set_index('SK_ID_CURR'), how='left', rsuffix='_pos').reset_index()
    df_fe = df_fe.set_index('SK_ID_CURR').join(install_payments_df_fe.set_index('SK_ID_CURR'), how='left', rsuffix='_ins').reset_index()
    df_fe = df_fe.set_index('SK_ID_CURR').join(credit_card_df_fe.set_index('SK_ID_CURR'), how='left', rsuffix='_cc').reset_index()

    del bureau_df_fe, prev_app_df_fe, pos_cash_df_fe, install_payments_df_fe, credit_card_df_fe
    gc.collect()

    # 4. Nettoyage des noms de colonnes et gestion des types/infinis (comme dans votre script MLflow)
    clean_column_mapping = {col: "".join(c if c.isalnum() else "_" for c in str(col)) for col in df_fe.columns}
    df_fe.rename(columns=clean_column_mapping, inplace=True)
    
    for col in df_fe.columns:
        df_fe[col] = pd.to_numeric(df_fe[col], errors='coerce')
    df_fe.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 5. Préparation des features pour le préprocesseur
    exclude_features_for_preprocessor = ['TARGET', 'SK_ID_CURR', 'index']
    features_for_preprocessor = [f for f in df_fe.columns if f not in exclude_features_for_preprocessor]
    
    # Créer un DataFrame avec uniquement les colonnes attendues par le préprocesseur, dans le bon ordre
    preprocessor_fitted_features = fitted_preprocessor.get_feature_names_out()
    preprocessor_fitted_features_cleaned = [name.replace('num__', '') for name in preprocessor_fitted_features]


    df_features_to_transform = pd.DataFrame(columns=preprocessor_fitted_features_cleaned, index=df_fe.index)
    
    for col in preprocessor_fitted_features_cleaned:
        if col in df_fe.columns:
            df_features_to_transform[col] = df_fe[col]
        else:
            df_features_to_transform[col] = np.nan # Remplir avec NaN si la colonne est manquante

    # 6. Application du préprocesseur ajusté
    if fitted_preprocessor is None:
        st.error("Le préprocesseur n'a pas pu être ajusté. Impossible de transformer les données.")
        return None

    # Appliquer la transformation. Le résultat est un tableau NumPy.
    transformed_array = fitted_preprocessor.transform(df_features_to_transform)
    
    # Convertir le tableau NumPy en DataFrame Pandas
    # Les noms de colonnes du DataFrame final doivent correspondre à `expected_features`
    # qui viennent de `input_example.json` du modèle MLflow.
    df_transformed_final = pd.DataFrame(transformed_array, columns=expected_features, index=raw_df_app.index)

    # Ré-ajouter SK_ID_CURR si vous l'avez retiré et qu'il est nécessaire pour l'output
    if sk_id_curr is not None and 'SK_ID_CURR' not in df_transformed_final.columns:
        df_transformed_final['SK_ID_CURR'] = sk_id_curr

    logger.info(f"Ingénierie des caractéristiques terminée dans Streamlit. Forme transformée finale: {df_transformed_final.shape}")
    return df_transformed_final


# --- Préparation des données d'entraînement (pour SHAP et Data Drift) ---

@st.cache_data(show_spinner="Chargement des données d'entraînement brutes pour SHAP et Data Drift depuis S3...")
def load_training_data_for_shap(file_key="input/application_train.csv"):
    """Charge les données d'entraînement brutes pour les calculs SHAP et Data Drift."""
    try:
        data = load_data_from_s3(file_key)
        if data is not None:
            if 'TARGET' in data.columns:
                data = data.drop(columns=['TARGET'], errors='ignore')
            logger.info("Données d'entraînement brutes pour SHAP et Data Drift chargées.")
            return data
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement des données d'entraînement brutes pour SHAP: {e}")
        logger.error(f"Error loading raw training data for SHAP: {e}")
        return None

# --- Préparation des données de référence pour SHAP (maintenant via FE Streamlit) ---
@st.cache_data(show_spinner="Préparation des données de référence pour SHAP...")
def prepare_shap_reference_data(fitted_preprocessor, expected_features, num_rows=None):
    """
    Charge les données d'entraînement brutes, applique le FE, et retourne un échantillon
    pour les calculs SHAP.
    """
    raw_data = load_training_data_for_shap(file_key="input/application_train.csv")
    if raw_data is None:
        return None
    
    # Appliquez le FE ici pour obtenir les données transformées pour SHAP
    transformed_data = run_feature_engineering_streamlit(raw_data, fitted_preprocessor, expected_features, is_training_data=True)

    if transformed_data is None:
        return None

    if num_rows:
        return transformed_data.sample(min(num_rows, len(transformed_data)), random_state=42)
    return transformed_data

# --- Explication SHAP ---

class IdentityPreprocessor:
    """Un préprocesseur factice qui ne fait rien. Utilisé quand les données sont déjà prétraitées."""
    def transform(self, X):
        return X
    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        # Fallback si X n'a pas de shape ou si input_features n'est pas fourni
        return [f"col_{i}" for i in range(X.shape[1])] if hasattr(X, 'shape') else []


@st.cache_resource(show_spinner="Calcul de l'explainer SHAP...")
def load_shap_explainer(_pyfunc_pipeline, all_training_features, fitted_preprocessor):
    """
    Charge l'explainer SHAP. Le préprocesseur est un IdentityPreprocessor car le FE est fait en amont.
    """
    preprocessor = IdentityPreprocessor()

    # Utilise les données transformées pour la référence SHAP
    ref_data_transformed = prepare_shap_reference_data(fitted_preprocessor, all_training_features, num_rows=1000)
    if ref_data_transformed is None:
        st.error("Impossible de charger les données de référence transformées pour l'explainer SHAP.")
        return None, None

    # S'assurer que seules les colonnes d'entraînement sont utilisées
    ref_data_transformed_filtered = ref_data_transformed[all_training_features]
    
    # Les données sont déjà transformées, le IdentityPreprocessor ne fait rien
    ref_data_processed_for_shap = preprocessor.transform(ref_data_transformed_filtered)
    
    # Les noms des features sont déjà les bons car ils proviennent du FE de Streamlit
    processed_feature_names = all_training_features 

    ref_data_df = pd.DataFrame(ref_data_processed_for_shap, columns=processed_feature_names)
    
    # Vérifier si le DataFrame de référence pour SHAP est vide
    if ref_data_df.empty:
        st.error("Le DataFrame de référence pour SHAP est vide. Impossible d'initialiser l'explainer SHAP.")
        logger.error("SHAP reference DataFrame is empty.")
        return None, None

    # Passer la fonction predict_proba du pipeline MLflow à SHAP.Explainer
    if not hasattr(_pyfunc_pipeline, 'predict_proba'):
        st.error("Le pipeline MLflow ne semble pas avoir une méthode 'predict_proba'. SHAP pourrait ne pas fonctionner.")
        logger.error("MLflow pipeline does not have 'predict_proba' method.")
        return None, None
        
    explainer = shap.Explainer(_pyfunc_pipeline.predict_proba, ref_data_df)
    return explainer, preprocessor


# --- Fonctions d'affichage ---

def plot_feature_importance(explainer, shap_values, feature_names, top_n=10):
    """Affiche les importances globales des caractéristiques SHAP."""
    if explainer is None or shap_values is None or not feature_names:
        st.warning("Impossible d'afficher l'importance des caractéristiques : données SHAP manquantes.")
        return

    # Si shap_values est une liste (pour classification multi-classes), prendre les valeurs de la classe positive (index 1)
    if isinstance(shap_values, list):
        # Assumer classe 1 pour classification binaire
        if len(shap_values) > 1:
            shap_values_abs_mean = np.abs(np.array(shap_values[1])).mean(0) 
        else: # Si une seule classe est retournée (ex: régression)
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
    
    # Pour la classification binaire, shap_values est une liste de 2 arrays. On prend le 2ème (classe 1)
    if isinstance(shap_values_individual, list) and len(shap_values_individual) > 1:
        shap_values_individual = shap_values_individual[1] # Prendre les valeurs pour la classe positive (1)

    try:
        # Expected value pour la classe 1 (positive)
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) and len(explainer.expected_value) > 1 else explainer.expected_value
        
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

# Chargement des données de test brutes (application_test.csv)
raw_test_data = load_data_from_s3("input/application_test.csv")
if raw_test_data is None:
    st.stop()

# Chargement du pipeline MLflow et de ses métadonnées et features d'entrée
pipeline, model_metadata, all_training_features = load_mlflow_pipeline_local()
if pipeline is None:
    st.stop()

# Vérifier si all_training_features a été récupéré du modèle
if not all_training_features:
    st.error("Impossible de récupérer les noms des caractéristiques d'entraînement depuis le modèle MLflow. Veuillez vous assurer que le modèle a été enregistré avec un `input_example.json` correct.")
    st.stop() # Arrêter l'exécution si les features ne peuvent pas être obtenues

# Récupérer le seuil optimal
threshold = 0.5
if model_metadata and 'optimal_threshold' in model_metadata:
    optimal_threshold_str = model_metadata['optimal_threshold']
    try:
        threshold = float(optimal_threshold_str)
        logger.info(f"Seuil optimal récupéré des métadonnées: {threshold}")
    except ValueError:
        logger.error(f"Erreur de conversion du seuil optimal: '{optimal_threshold_str}'. Utilisation de 0.5.")
        threshold = 0.5
else:
    logger.warning("Seuil optimal non trouvé dans les métadonnées du modèle. Utilisation de 0.5 par défaut.")
    threshold = 0.5

# Récupérer les informations sur les features pour Streamlit
features_info_for_streamlit = {}
if model_metadata and 'features_info_for_streamlit' in model_metadata:
    features_info_json = model_metadata['features_info_for_streamlit']
    try:
        features_info_for_streamlit = json.loads(features_info_json)
        logger.info("Informations sur les features pour Streamlit récupérées des métadonnées.")
    except json.JSONDecodeError:
        logger.error(f"Erreur de décodage JSON pour 'features_info_for_streamlit': {features_info_json}. Utilisation d'un dictionnaire vide.")
        features_info_for_streamlit = {}
else:
    logger.warning("Informations sur les features pour Streamlit non trouvées dans les métadonnées. Utilisation d'un dictionnaire vide.")
    features_info_for_streamlit = {}


# Assurez-vous que 'SK_ID_CURR' n'est JAMAIS dans les features d'entraînement
if 'SK_ID_CURR' in all_training_features:
    all_training_features.remove('SK_ID_CURR')
    logger.info("SK_ID_CURR retiré de all_training_features.")

# Obtenir le préprocesseur ajusté sur les données d'entraînement
fitted_preprocessor = get_fitted_preprocessor()
if fitted_preprocessor is None:
    st.error("Impossible d'initialiser le préprocesseur. L'application ne peut pas continuer.")
    st.stop()
    
# Exécution du Feature Engineering sur les données de test réelles (avec les features attendues)
data_transformed = run_feature_engineering_streamlit(raw_test_data, fitted_preprocessor, all_training_features, is_training_data=False)
if data_transformed is None:
    st.error("Échec de l'ingénierie des caractéristiques pour les données de test.")
    st.stop()

# Vérifier que les colonnes nécessaires sont présentes dans 'data_transformed'
# et que l'ordre des colonnes correspond à celui attendu par le modèle MLflow.
missing_features_in_data = [f for f in all_training_features if f not in data_transformed.columns]
if missing_features_in_data:
    st.warning(f"Attention : Les caractéristiques suivantes du modèle sont manquantes dans les données de test transformées : {', '.join(missing_features_in_data)}. Le modèle pourrait ne pas fonctionner comme prévu.")
    # Filtrer all_training_features pour ne garder que celles qui sont dans `data_transformed`
    all_training_features = [f for f in all_training_features if f in data_transformed.columns]

# Si des colonnes sont manquantes après le filtrage, cela signifie que votre FE dans Streamlit
# ne produit pas toutes les colonnes attendues par le modèle.
if not all_training_features:
    st.error("Aucune caractéristique d'entraînement valide n'a été trouvée après l'ingénierie des caractéristiques et la validation. Veuillez vérifier votre fonction `run_feature_engineering_streamlit` et les métadonnées du modèle.")
    st.stop()


# --- Sidebar pour la sélection du client ---
st.sidebar.header("Sélection du Client")
# Vérifier si 'SK_ID_CURR' est dans data_transformed avant de l'utiliser
if 'SK_ID_CURR' not in data_transformed.columns:
    st.error("La colonne 'SK_ID_CURR' est manquante dans les données transformées. Impossible de sélectionner un client.")
    st.stop()

client_ids = data_transformed['SK_ID_CURR'].tolist() 
selected_client_id = st.sidebar.selectbox("Sélectionnez un ID Client :", client_ids)

# Trouver les données du client sélectionné dans le DataFrame transformé
client_data_transformed_row = data_transformed[data_transformed['SK_ID_CURR'] == selected_client_id].iloc[0]

# Préparer le DataFrame pour la prédiction
# Assurez-vous que l'ordre des colonnes correspond à all_training_features
client_data_for_prediction = client_data_transformed_row[all_training_features].to_frame().T

# Log pour le débogage :
logger.info(f"Features passées au pipeline: {all_training_features}")
logger.info(f"Colonnes disponibles dans client_data_for_prediction: {client_data_for_prediction.columns.tolist()}")
logger.info(f"Forme de client_data_for_prediction: {client_data_for_prediction.shape}")


# --- Prédiction ---
with st.spinner("Calcul de la prédiction..."):
    try:
        prediction_proba = pipeline.predict(client_data_for_prediction)[0]
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

# Charger l'explainer SHAP et le préprocesseur (IdentityPreprocessor)
explainer, preprocessor = load_shap_explainer(pipeline, all_training_features, fitted_preprocessor)

if explainer is not None and preprocessor is not None:
    try:
        # Les données pour SHAP sont les mêmes que celles pour la prédiction
        shap_input_data = client_data_for_prediction
        processed_feature_names = all_training_features

        with st.spinner("Calcul des valeurs SHAP..."):
            shap_values = explainer.shap_values(shap_input_data)
        
        st.write("Les valeurs SHAP montrent l'impact de chaque caractéristique sur la prédiction du modèle.")
        
        if prediction_proba is not None:
            if isinstance(shap_values, list) and len(shap_values) > 1:
                individual_shap_values = shap_values[1][0] # Pour la classe 1, et le premier échantillon
            else:
                individual_shap_values = shap_values[0] # Si c'est un tableau unique (régression ou 1D classification)
            
            client_processed_df = pd.DataFrame(shap_input_data, columns=processed_feature_names)

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
