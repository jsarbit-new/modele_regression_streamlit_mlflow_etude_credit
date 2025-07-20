# app.py

import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import logging
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import os
import gc

# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration MLflow pour le client Streamlit (Adaptée pour exécution locale) ---
# IMPORTANT : Pour le déploiement sur Streamlit Cloud, vous devrez remettre les st.secrets
# et configurer vos secrets dans le tableau de bord Streamlit Cloud.
# Pour l'exécution locale, nous utilisons l'adresse IP publique de votre EC2 et
# nous nous attendons à ce que les variables d'environnement AWS soient définies localement.

MLFLOW_MODEL_NAME = "HomeCreditLogisticRegressionPipelineStubs" # Nom du modèle enregistré
MLFLOW_MODEL_STAGE = "Production" # Ou "Staging", "None" pour la dernière version

# Adresse IP publique de votre instance EC2 où tourne le serveur MLflow
# Remplacez "16.170.254.32" si votre IP publique EC2 a changé.
EC2_PUBLIC_IP = "16.170.254.32" 
MLFLOW_TRACKING_URI = f"http://{EC2_PUBLIC_IP}:5000"

# Votre bucket S3 et région (doivent correspondre à votre setup AWS)
AWS_S3_BUCKET = "modele-regression-streamlit-mlflow-etude-credit" 
AWS_REGION = "eu-north-1" # La région de votre bucket S3

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Pour l'exécution locale, assurez-vous que ces variables sont définies dans votre terminal
    # AVANT de lancer l'application Streamlit.
    # Ex: (dans votre Miniconda Prompt)
    # set AWS_ACCESS_KEY_ID=VOTRE_CLE_ACCES_AWS
    # set AWS_SECRET_ACCESS_KEY=VOTRE_CLE_SECRETE_AWS
    # set AWS_REGION=eu-north-1
    # set MLFLOW_S3_ENDPOINT_URL=https://s3.eu-north-1.amazonaws.com
    
    # Vérification si les variables d'environnement AWS sont définies
    if "AWS_ACCESS_KEY_ID" not in os.environ or \
       "AWS_SECRET_ACCESS_KEY" not in os.environ or \
       "AWS_REGION" not in os.environ:
        st.error("Les variables d'environnement AWS (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION) ne sont pas définies. "
                 "Veuillez les définir dans votre terminal avant de lancer l'application Streamlit.")
        st.stop() # Arrête l'exécution de l'application

    # Définition de MLFLOW_S3_ENDPOINT_URL si elle n'est pas déjà définie, en utilisant la région
    if "MLFLOW_S3_ENDPOINT_URL" not in os.environ:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"https://s3.{AWS_REGION}.amazonaws.com"

    logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
    logger.info(f"AWS S3 Bucket: {AWS_S3_BUCKET}, Region: {AWS_REGION}")
    
except Exception as e:
    st.error(f"Erreur de configuration MLflow/AWS : {e}. Assurez-vous que l'URI de tracking et l'accès S3 sont corrects.")
    logger.error(f"MLflow configuration error: {e}", exc_info=True)
    st.stop()

# --- FONCTIONS STUBS POUR LE FEATURE ENGINEERING CÔTÉ STREAMLIT ---
# Ces fonctions DOIVENT être cohérentes avec celles du script MLflow pour les noms de colonnes et types.
# Elles simulent la création des features.
def load_application_data_stub_streamlit(num_rows=1, features_info_for_streamlit_ref=None):
    """Simule le chargement de données d'application pour Streamlit (une seule ligne)."""
    data = {}
    # Générer des features numériques basées sur la configuration chargée du modèle
    for feature_name, info in (features_info_for_streamlit_ref or {}).items():
        if info.get("type") == "numerical" and feature_name.startswith('app_feature_'):
            min_v, max_v = info["min_val"], info["max_val"]
            data[feature_name] = [np.random.rand() * (max_v - min_v) + min_v]
        elif info.get("type") == "categorical" and feature_name.startswith('app_feature_'):
            data[feature_name] = [np.random.choice(info.get("options", ["CategoryA", "CategoryB"]))]
    
    # Ajouter des features numériques génériques supplémentaires si nécessaire (si non spécifiées dans features_info)
    # Cette logique doit être robuste pour inclure toutes les colonnes que le modèle s'attend à voir après le FE.
    # Pour la démo avec des stubs, on génère juste des noms génériques.
    for i in range(50): # S'assurer d'avoir au moins 50 app_features simulées
        feat_name = f'app_feature_{i}'
        if feat_name not in data:
            data[feat_name] = [np.random.rand() * 100]

    df = pd.DataFrame(data)
    df['SK_ID_CURR'] = [0] # ID client simulé
    
    # Ajouter des colonnes catégorielles pour simuler, si elles ne sont pas déjà dans les sliders
    if 'NAME_CONTRACT_TYPE' not in df.columns:
        df['NAME_CONTRACT_TYPE'] = ['Cash']
    if 'CODE_GENDER' not in df.columns:
        df['CODE_GENDER'] = ['M']
    if 'FLAG_OWN_CAR' not in df.columns:
        df['FLAG_OWN_CAR'] = ['Y']
    if 'NAME_INCOME_TYPE' not in df.columns:
        df['NAME_INCOME_TYPE'] = ['Working']
    
    # Simuler quelques NaN pour tester l'imputation
    if 'app_feature_0' in df.columns:
        df['app_feature_0'].iloc[0] = np.nan 
    else:
        df['another_feature_with_nan'] = np.nan

    return df

def process_bureau_data_stub_streamlit(df_app_main, features_info_for_streamlit_ref=None):
    """Simule l'ajout de features agrégées de bureau pour Streamlit."""
    if 'bureau_debt_ratio' not in df_app_main.columns:
        info = (features_info_for_streamlit_ref or {}).get('bureau_debt_ratio', {"min_val": 0.0, "max_val": 5.0})
        df_app_main['bureau_debt_ratio'] = [np.random.rand() * (info["max_val"] - info["min_val"]) + info["min_val"]]
    
    if 'bureau_active_loans_perc' not in df_app_main.columns:
        info = (features_info_for_streamlit_ref or {}).get('bureau_active_loans_perc', {"min_val": 0.0, "max_val": 1.0})
        df_app_main['bureau_active_loans_perc'] = [np.random.rand() * (info["max_val"] - info["min_val"]) + info["min_val"]]

    for i in range(5):
        if f'bureau_feat_gen_{i}' not in df_app_main.columns:
            df_app_main[f'bureau_feat_gen_{i}'] = [np.random.rand() * 10]
    return df_app_main

def process_previous_applications_data_stub_streamlit(df_app_main, features_info_for_streamlit_ref=None):
    """Simule l'ajout de features agrégées de précédentes applications pour Streamlit."""
    if 'prev_app_credit_ratio' not in df_app_main.columns:
        info = (features_info_for_streamlit_ref or {}).get('prev_app_credit_ratio', {"min_val": 0.0, "max_val": 1.0})
        df_app_main['prev_app_credit_ratio'] = [np.random.rand() * (info["max_val"] - info["min_val"]) + info["min_val"]]
    
    if 'prev_app_refused_perc' not in df_app_main.columns:
        info = (features_info_for_streamlit_ref or {}).get('prev_app_refused_perc', {"min_val": 0.0, "max_val": 1.0})
        df_app_main['prev_app_refused_perc'] = [np.random.rand() * (info["max_val"] - info["min_val"]) + info["min_val"]]

    for i in range(5):
        if f'prev_app_feat_gen_{i}' not in df_app_main.columns:
            df_app_main[f'prev_app_feat_gen_{i}'] = [np.random.rand() * 5]
    return df_app_main

def process_pos_cash_data_stub_streamlit(df_app_main, features_info_for_streamlit_ref=None):
    for i in range(3):
        if f'pos_feat_gen_{i}' not in df_app_main.columns:
            df_app_main[f'pos_feat_gen_{i}'] = [np.random.rand() * 10]
    return df_app_main

def process_installments_payments_data_stub_streamlit(df_app_main, features_info_for_streamlit_ref=None):
    for i in range(3):
        if f'install_feat_gen_{i}' not in df_app_main.columns:
            df_app_main[f'install_feat_gen_{i}'] = [np.random.rand() * 20]
    return df_app_main

def process_credit_card_balance_data_stub_streamlit(df_app_main, features_info_for_streamlit_ref=None):
    for i in range(3):
        if f'cc_feat_gen_{i}' not in df_app_main.columns:
            df_app_main[f'cc_feat_gen_{i}'] = [np.random.rand() * 50]
    return df_app_main


def run_feature_engineering_streamlit(client_input_df_original=None, features_info_for_streamlit_ref=None):
    """
    Exécute la chaîne de fonctions d'ingénierie des caractéristiques (stubs) pour une seule ligne.
    Prend un DataFrame 'original' (par ex. du slider ou de l'échantillon de test)
    et génère un DataFrame complet avec toutes les features simulées.
    """
    logger.info("## Début du FE Stubs pour Streamlit.")
    
    # Créer un DataFrame de base avec les colonnes de l'application et de l'ID client
    # C'est ici que nous injectons les valeurs des sliders.
    if client_input_df_original is None:
        # Si aucun input n'est fourni, générer une ligne de base
        df_base = load_application_data_stub_streamlit(num_rows=1, features_info_for_streamlit_ref=features_info_for_streamlit_ref)
    else:
        df_base = client_input_df_original.copy()
        # Assurez-vous que df_base contient les colonnes catégorielles attendues si elles ne sont pas dans les sliders
        for col_cat in ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_INCOME_TYPE']:
            if col_cat not in df_base.columns:
                # Ajouter une valeur par défaut cohérente ou aléatoire
                df_base[col_cat] = np.random.choice(['Cash', 'Revolving', 'M', 'F', 'Y', 'N', 'Working'], len(df_base)) # Exemple générique
        
        # Compléter avec les app_features non définies par les sliders
        for i in range(50): 
            feat_name = f'app_feature_{i}'
            if feat_name not in df_base.columns:
                info = (features_info_for_streamlit_ref or {}).get(feat_name, {"min_val": 0.0, "max_val": 1.0, "type": "numerical"})
                if info.get("type") == "numerical":
                    min_v, max_v = info["min_val"], info["max_val"]
                    df_base[feat_name] = [np.random.rand() * (max_v - min_v) + min_v]
                elif info.get("type") == "categorical":
                    df_base[feat_name] = [np.random.choice(info.get("options", ["CategoryA"]))]


    df = process_bureau_data_stub_streamlit(df_base, features_info_for_streamlit_ref)
    df = process_previous_applications_data_stub_streamlit(df, features_info_for_streamlit_ref)
    df = process_pos_cash_data_stub_streamlit(df, features_info_for_streamlit_ref)
    df = process_installments_payments_data_stub_streamlit(df, features_info_for_streamlit_ref)
    df = process_credit_card_balance_data_stub_streamlit(df, features_info_for_streamlit_ref)
    
    # Nettoyage et typage (important pour la cohérence avec le pipeline sklearn)
    for col in df.columns:
        if col != 'SK_ID_CURR': # SK_ID_CURR est un identifiant, pas une feature pour le modèle
            if df[col].dtype == object:
                df[col] = df[col].astype('category')
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True) # Remplacer inf par NaN
    
    logger.info(f"[Feature Engineering Stubs Streamlit] terminé. Forme finale : {df.shape}")
    return df

# --- Chargement du pipeline MLflow Scikit-learn ---
@st.cache_resource(show_spinner="Chargement du modèle MLflow...")
def load_mlflow_pipeline():
    try:
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
        logger.info(f"Chargement du modèle MLflow depuis : {model_uri}")
        pipeline = mlflow.sklearn.load_model(model_uri)
        logger.info("Modèle MLflow (pipeline Scikit-learn) chargé avec succès.")

        # Récupérer les métadonnées du modèle pour la configuration des features
        client = mlflow.tracking.MlflowClient()
        # On cherche la version la plus récente dans le stage 'Production'
        model_versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
        latest_prod_version = None
        for mv in model_versions:
            if MLFLOW_MODEL_STAGE in mv.current_stage:
                if latest_prod_version is None or mv.version > latest_prod_version.version:
                    latest_prod_version = mv
        
        if latest_prod_version is None:
            st.error(f"Aucune version du modèle '{MLFLOW_MODEL_NAME}' trouvée dans le stage '{MLFLOW_MODEL_STAGE}'.")
            return None, 0.5, {}, []

        model_metadata = latest_prod_version.tags if latest_prod_version.tags else {}

        features_info_for_streamlit = {}
        if "features_info_for_streamlit" in model_metadata:
            try:
                features_info_for_streamlit = json.loads(model_metadata["features_info_for_streamlit"])
            except json.JSONDecodeError:
                logger.error("Erreur de décodage JSON pour features_info_for_streamlit dans les métadonnées.")
        
        all_training_features_names = []
        if "all_training_features" in model_metadata:
            try:
                all_training_features_names = json.loads(model_metadata["all_training_features"])
            except json.JSONDecodeError:
                logger.error("Erreur de décodage JSON pour all_training_features dans les métadonnées.")

        optimal_threshold = float(model_metadata.get("optimal_threshold", 0.5))

        logger.info(f"Modèle chargé. Seuil optimal: {optimal_threshold}, Features infos: {len(features_info_for_streamlit)}, All model features: {len(all_training_features_names)}")

        return pipeline, optimal_threshold, features_info_for_streamlit, all_training_features_names

    except Exception as e:
        st.error(f"Impossible de charger le pipeline MLflow : {e}. Vérifiez l'URI de tracking et les permissions S3.")
        logger.error(f"Error loading MLflow pipeline: {e}", exc_info=True)
        return None, 0.5, {}, []

# --- Préparation des données de référence pour SHAP ---
@st.cache_data(show_spinner="Préparation des données de référence pour SHAP...")
def prepare_shap_reference_data(mlflow_pipeline, features_for_streamlit_config, all_model_features, num_rows=200):
    """
    Génère un échantillon de données via les stubs pour SHAP, puis le transforme avec le préprocesseur du modèle.
    """
    # Générer un échantillon de données en utilisant les stubs de FE (comme l'entraînement)
    # Pour cela, on va simuler l'input des sliders pour chaque ligne de l'échantillon
    sample_data_list = []
    for _ in range(num_rows):
        single_row_input = {}
        for feature_name, info in features_for_streamlit_config.items():
            if info.get("type") == "numerical":
                min_v, max_v = info["min_val"], info["max_val"]
                single_row_input[feature_name] = np.random.rand() * (max_v - min_v) + min_v
            elif info.get("type") == "categorical":
                single_row_input[feature_name] = np.random.choice(info.get("options", ["CategoryA", "CategoryB"]))
        sample_data_list.append(single_row_input)
    
    sample_df_from_sliders_simulated = pd.DataFrame(sample_data_list)
    
    # Passer par la fonction de FE complète des stubs pour générer toutes les features
    sample_df_fe_stubs = run_feature_engineering_streamlit(sample_df_from_sliders_simulated, features_for_streamlit_config)
    
    # S'assurer que le DataFrame a les mêmes colonnes que celles attendues par le modèle
    # et dans le même ordre.
    # Les colonnes manquantes seront remplies avec NaN (le preprocessor gérera l'imputation).
    # Les colonnes en trop seront ignorées par le ColumnTransformer.
    processed_df_for_sklearn = pd.DataFrame(columns=all_model_features, index=sample_df_fe_stubs.index)
    for col in all_model_features:
        if col in sample_df_fe_stubs.columns:
            processed_df_for_sklearn[col] = sample_df_fe_stubs[col]
        else:
            processed_df_for_sklearn[col] = np.nan # NaN pour les colonnes manquantes
            
    # S'assurer que les types catégoriels sont corrects avant le preprocessor
    for col in processed_df_for_sklearn.columns:
        # Check if the feature is designated as categorical by features_info_for_streamlit_config
        if col in features_for_streamlit_config and features_for_streamlit_config[col].get('type') == 'categorical':
            processed_df_for_sklearn[col] = processed_df_for_sklearn[col].astype('category')
        elif pd.api.types.is_numeric_dtype(processed_df_for_sklearn[col]):
            processed_df_for_sklearn[col] = pd.to_numeric(processed_df_for_sklearn[col], errors='coerce')


    # Appliquer le préprocesseur du pipeline sklearn pour obtenir les données au format SHAP
    preprocessor = mlflow_pipeline.named_steps['preprocessor']
    transformed_data_for_explainer = preprocessor.transform(processed_df_for_sklearn)

    # Récupérer les noms de features post-transformation pour SHAP
    feature_names_for_shap = preprocessor.get_feature_names_out() # get_feature_names_out() est la bonne méthode

    # Convertir en DataFrame pour SHAP
    final_shap_ref_data = pd.DataFrame(transformed_data_for_explainer, columns=feature_names_for_shap)
    
    logger.info(f"Données de référence SHAP préparées. Forme: {final_shap_ref_data.shape}")
    gc.collect() # Nettoyage mémoire
    return final_shap_ref_data, feature_names_for_shap

# --- Explication SHAP ---
@st.cache_resource(show_spinner="Calcul de l'explainer SHAP...")
def load_shap_explainer(mlflow_pipeline, features_info_for_streamlit_ref, all_model_features):
    """
    Charge l'explainer SHAP en utilisant le classifieur interne du modèle MLflow.
    """
    ref_data_transformed, feature_names_for_shap = prepare_shap_reference_data(mlflow_pipeline, features_info_for_streamlit_ref, all_model_features, num_rows=100) # Petit échantillon pour SHAP
    
    if ref_data_transformed is None or ref_data_transformed.empty:
        st.error("Impossible de préparer les données de référence transformées pour l'explainer SHAP. Le DataFrame est vide.")
        return None, None

    classifier = mlflow_pipeline.named_steps['classifier']
    
    try:
        explainer = shap.Explainer(classifier.predict_proba, ref_data_transformed)
        logger.info("Explainer SHAP chargé avec succès.")
        return explainer, feature_names_for_shap
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de l'explainer SHAP : {e}. Vérifiez la compatibilité du modèle et des données de référence.")
        logger.error(f"Error initializing SHAP explainer: {e}", exc_info=True)
        return None, None

# --- Fonctions d'affichage SHAP ---
def plot_feature_importance(explainer, shap_values, feature_names, top_n=10):
    """Affiche les importances globales des caractéristiques SHAP."""
    if explainer is None or shap_values is None or not feature_names:
        st.warning("Impossible d'afficher l'importance des caractéristiques : données SHAP manquantes.")
        return

    # Si shap_values est une liste (pour classification multi-classes), prendre les valeurs de la classe positive (index 1)
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            shap_values_abs_mean = np.abs(np.array(shap_values[1])).mean(0)    
        else: # Cas où il n'y a qu'une seule classe retournée (ex: régression)
            shap_values_abs_mean = np.abs(np.array(shap_values[0])).mean(0)
    else: # Si shap_values est directement un array (pour régression par ex.)
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
        
    if isinstance(shap_values_individual, list) and len(shap_values_individual) > 1:
        shap_values_individual = shap_values_individual[1] # Prendre les SHAP values pour la classe positive

    try:
        # Expected value pour la classe positive
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) and len(explainer.expected_value) > 1 else explainer.expected_value
            
        # Assurez-vous que processed_features_df est un DataFrame avec une seule ligne et les bonnes colonnes
        if not isinstance(processed_features_df, pd.DataFrame):
            # C'est un array numpy ici si transformé par le préprocesseur
            processed_features_df_for_plot = pd.DataFrame(processed_features_df.reshape(1, -1), columns=feature_names)
        elif processed_features_df.shape[0] > 1:
            processed_features_df_for_plot = processed_features_df.iloc[0:1] # Ne prendre que la première ligne
        else:
            processed_features_df_for_plot = processed_features_df

        # S'assurer que les colonnes sont dans le bon ordre pour SHAP
        processed_features_df_for_plot = processed_features_df_for_plot[feature_names]

        shap.initjs()
            
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_individual, 
                base_values=expected_value, 
                data=processed_features_df_for_plot.iloc[0].values, # S'assurer que c'est un array 1D
                feature_names=feature_names
            ),
            max_display=20, # Afficher les 20 features les plus influentes
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

# Chargement du pipeline MLflow Scikit-learn et de ses métadonnées
mlflow_pipeline, threshold, features_info_for_streamlit, all_model_features = load_mlflow_pipeline()

if mlflow_pipeline is None:
    st.error("Impossible de charger le pipeline MLflow. L'application ne peut pas démarrer.")
    st.stop()

# --- Sidebar pour la sélection du client ou l'entrée manuelle ---
st.sidebar.header("Sélection du Client ou Entrée Manuelle")

# On charge des données de test brutes, juste pour simuler la sélection d'un client ID
# Ces données ne sont PAS passées directement au modèle, elles servent juste à l'ID client.
@st.cache_data(show_spinner="Chargement des ID clients de test...")
def load_test_ids():
    # Puisque nous n'avons pas de "vrai" fichier test.csv sur S3 dans cette approche,
    # nous allons simuler les ID clients pour la démo.
    return pd.DataFrame({'SK_ID_CURR': np.arange(1000, 1050).tolist()})

raw_test_ids = load_test_ids()
client_ids = raw_test_ids['SK_ID_CURR'].tolist()    
selected_client_id = st.sidebar.selectbox("Sélectionnez un ID Client :", client_ids)

st.sidebar.subheader("Entrée Manuelle des Caractéristiques (Simulées)")
# Les sliders pour les features importantes
user_input_features = {}
st.sidebar.write("Ajustez les valeurs des caractéristiques simulées :")
numeric_features_to_display = {k: v for k, v in features_info_for_streamlit.items() if v.get("type") == "numerical"}
categorical_features_to_display = {k: v for k, v in features_info_for_streamlit.items() if v.get("type") == "categorical"}


# Sliders pour les features numériques
for feature_name, info in numeric_features_to_display.items():
    if feature_name.startswith('app_feature_') or feature_name.startswith('bureau_'): # Filtrer pour ne pas surcharger
        current_value = info["min_val"] + (info["max_val"] - info["min_val"]) / 2 # Valeur par défaut au milieu
        user_input_features[feature_name] = st.sidebar.slider(
            info["display_name"],
            float(info["min_val"]),
            float(info["max_val"]),
            float(current_value),
            key=f"slider_{feature_name}"
        )

# Selectbox pour les features catégorielles (si vous les gérez manuellement)
for feature_name, info in categorical_features_to_display.items():
    if feature_name.startswith('NAME_') or feature_name.startswith('CODE_'): # Exemples
        options = info.get("options", ["Option1", "Option2"])
        user_input_features[feature_name] = st.sidebar.selectbox(
            info["display_name"],
            options,
            key=f"selectbox_{feature_name}"
        )


# Créer un DataFrame d'entrée pour le FE à partir des sliders
client_input_df_from_sliders = pd.DataFrame([user_input_features])
client_input_df_from_sliders['SK_ID_CURR'] = selected_client_id # Ajouter l'ID client pour la référence


# --- Prédiction ---
with st.spinner("Calcul de la prédiction..."):
    try:
        # Appliquer le FE (stubs) pour obtenir les features au format attendu par le modèle
        # Cela inclura les colonnes des sliders et toutes les autres colonnes générées par les stubs
        processed_client_data_for_prediction = run_feature_engineering_streamlit(
            client_input_df_from_sliders, features_info_for_streamlit
        )
        
        # S'assurer que les colonnes sont dans le bon ordre et que les colonnes manquantes sont gérées.
        # Le ColumnTransformer du pipeline sklearn est sensible à l'ordre des colonnes.
        # Nous devons fournir un DataFrame avec EXACTEMENT les mêmes colonnes que celles utilisées
        # pour l'entraînement (all_model_features), dans le bon ordre.
        final_input_for_model = pd.DataFrame(columns=all_model_features, index=processed_client_data_for_prediction.index)
        for col in all_model_features:
            if col in processed_client_data_for_prediction.columns:
                final_input_for_model[col] = processed_client_data_for_prediction[col]
            else:
                final_input_for_model[col] = np.nan # Gérer les colonnes manquantes (seront imputées par le modèle)
        
        # Assurer les types de données cohérents
        for col in final_input_for_model.columns:
            # Check if the feature is designated as numerical by features_info_for_streamlit_config
            if col in features_info_for_streamlit and features_info_for_streamlit[col].get('type') == 'numerical':
                final_input_for_model[col] = pd.to_numeric(final_input_for_model[col], errors='coerce')
            # Check if the feature is designated as categorical by features_info_for_streamlit_config
            elif col in features_info_for_streamlit and features_info_for_streamlit[col].get('type') == 'categorical':
                final_input_for_model[col] = final_input_for_model[col].astype('category')
            # Fallback for other columns not explicitly listed in features_info_for_streamlit_config
            # but expected by the model's preprocessor.
            # This logic depends on your preprocessor. Here we assume numeric as default for other cols.
            else: 
                final_input_for_model[col] = pd.to_numeric(final_input_for_model[col], errors='coerce')


        if final_input_for_model.empty or final_input_for_model.shape[0] == 0:
            st.error("Les données du client après FE sont vides. Impossible de prédire.")
            prediction_proba = None
            prediction = None
        else:
            prediction_proba = mlflow_pipeline.predict_proba(final_input_for_model)[:, 1][0]
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

# Charger l'explainer SHAP et les noms de features post-ColumnTransformer
explainer, shap_feature_names = load_shap_explainer(mlflow_pipeline, features_info_for_streamlit, all_model_features)

if explainer is not None and shap_feature_names is not None:
    try:
        with st.spinner("Calcul des valeurs SHAP pour le client sélectionné..."):
            # Les données d'entrée pour SHAP doivent être exactement comme celles qui sortent du préprocesseur.
            # On applique les stubs au client_input_df_from_sliders pour obtenir les features ingéniérisées
            processed_client_data_for_shap = run_feature_engineering_streamlit(
                client_input_df_from_sliders, features_info_for_streamlit
            )
            
            # Puis on s'assure qu'elles sont dans le bon ordre et que les manquantes sont gérées
            final_input_for_shap_preprocessor = pd.DataFrame(columns=all_model_features, index=processed_client_data_for_shap.index)
            for col in all_model_features:
                if col in processed_client_data_for_shap.columns:
                    final_input_for_shap_preprocessor[col] = processed_client_data_for_shap[col]
                else:
                    final_input_for_shap_preprocessor[col] = np.nan
            
            # Assurer les types de données cohérents
            for col in final_input_for_shap_preprocessor.columns:
                if col in features_info_for_streamlit and features_info_for_streamlit[col].get('type') == 'numerical':
                    final_input_for_shap_preprocessor[col] = pd.to_numeric(final_input_for_shap_preprocessor[col], errors='coerce')
                elif col in features_info_for_streamlit and features_info_for_streamlit[col].get('type') == 'categorical':
                    final_input_for_shap_preprocessor[col] = final_input_for_shap_preprocessor[col].astype('category')
                else:
                    final_input_for_shap_preprocessor[col] = pd.to_numeric(final_input_for_shap_preprocessor[col], errors='coerce')


            # Appliquer la transformation du ColumnTransformer
            shap_input_data_transformed_array = mlflow_pipeline.named_steps['preprocessor'].transform(final_input_for_shap_preprocessor)
            
            # Calculer les valeurs SHAP
            shap_values = explainer.shap_values(shap_input_data_transformed_array)
                
        st.write("Les valeurs SHAP montrent l'impact de chaque caractéristique sur la prédiction du modèle.")
            
        if prediction_proba is not None: # Vérifier si la prédiction a réussi avant d'afficher SHAP
            plot_individual_explanation(explainer, shap_values, shap_input_data_transformed_array, shap_feature_names, selected_client_id)
            gc.collect() # Nettoyage mémoire après le tracé SHAP
        else:
            st.warning("Impossible de générer l'explication SHAP car la prédiction a échoué.")

    except Exception as e:
        st.error(f"Erreur lors du calcul ou de l'affichage des valeurs SHAP : {e}")
        logger.error(f"SHAP calculation/plotting error: {e}", exc_info=True)