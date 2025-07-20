import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, fbeta_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import json
import logging
import os
import gc # Pour la gestion de la mémoire

# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration MLflow ---
<<<<<<< HEAD
# IMPORTANT : Remplacez "16.170.254.32" par l'adresse IP PUBLIQUE de votre instance EC2
# Si votre IP change, vous devrez mettre à jour cette ligne ici ET la variable d'environnement locale.
MLFLOW_TRACKING_URI = "http://16.170.254.32:5000"

# Nom de votre bucket S3 que vous avez créé (aucune modification nécessaire si c'est correct)
AWS_S3_BUCKET = "modele-regression-streamlit-mlflow-etude-credit" 

# La région de votre bucket S3 (aucune modification nécessaire si c'est correct)
AWS_REGION = "eu-north-1" 

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Les variables d'environnement AWS (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION)
    # ainsi que MLFLOW_S3_ENDPOINT_URL (s3://<YOUR_S3_BUCKET_NAME>)
    # DOIVENT ÊTRE DÉFINIES DANS VOTRE TERMINAL LOCAL AVANT DE LANCER CE SCRIPT.
    # Ex:
    # set MLFLOW_TRACKING_URI=http://16.170.254.32:5000
    # set MLFLOW_S3_ENDPOINT_URL=s3://modele-regression-streamlit-mlflow-etude-credit
    # set AWS_ACCESS_KEY_ID=VOTRE_CLE_ACCES_AWS
    # set AWS_SECRET_ACCESS_KEY=VOTRE_CLE_SECRETE_AWS
    # set AWS_DEFAULT_REGION=eu-north-1
    #
    # Ces lignes sont commentées car elles ne devraient pas être nécessaires si vous les définissez en amont.
    # os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") 
    # os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
    # os.environ["AWS_REGION"] = AWS_REGION
    # os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"https://s3.{AWS_REGION}.amazonaws.com" # Ou simplement s3://<BUCKET_NAME>
=======
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" # URI de votre serveur MLflow
AWS_S3_BUCKET = "modele-regression-streamlit-mlflow-etude-credit" # Votre bucket S3
AWS_REGION = "eu-north-1" # La région de votre bucket S3

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Configure l'accès S3 pour MLflow. C'est crucial pour l'enregistrement des artefacts.
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") # Assurez-vous que ces variables sont définies dans votre environnement
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
    os.environ["AWS_REGION"] = AWS_REGION
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"https://s3.{AWS_REGION}.amazonaws.com" # Point d'accès S3 spécifique à la région
>>>>>>> f19f2d7b0059efff3b5e238d74411ad289f4909f

    logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
    logger.info(f"AWS S3 Bucket: {AWS_S3_BUCKET}, Region: {AWS_REGION}")
except Exception as e:
<<<<<<< HEAD
    logger.error(f"Failed to set MLflow tracking URI: {e}")
=======
    logger.error(f"Failed to set MLflow tracking URI or AWS environment variables: {e}")
>>>>>>> f19f2d7b0059efff3b5e238d74411ad289f4909f
    logger.error("MLflow tracking and S3 artifact storage might not work correctly.")
    exit(1)

MLFLOW_EXPERIMENT_NAME = "Home_Credit_Default_Risk_Logistic_Regression_Pipeline_Stubs"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

MLFLOW_MODEL_NAME = "HomeCreditLogisticRegressionPipelineStubs" # Nom du modèle enregistré

CUSTOM_FBETA_BETA = np.sqrt(10)
fbeta_scorer = make_scorer(fbeta_score, beta=CUSTOM_FBETA_BETA, average='binary', pos_label=1)
logger.info(f"Le seuil de classification sera optimisé pour le F-beta score (beta={CUSTOM_FBETA_BETA}).")

# --- MAPPING SHAP IMPORTANT FEATURES INFO ---
# TRÈS IMPORTANT : Ce dictionnaire DOIT refléter les NOMS DE COLONNES et les PLAGES DE VALEURS
# générés par VOS FONCTIONS STUBS. C'est ce que Streamlit utilisera pour les sliders.
SHAP_IMPORTANT_FEATURES_INFO = {
    # App features (simulées)
    "app_feature_0": {"display_name": "Taux Population Région", "min_val": 0.001, "max_val": 0.1, "type": "numerical"},
    "app_feature_1": {"display_name": "Ratio Annuité/Revenu", "min_val": 0.01, "max_val": 1.5, "type": "numerical"},
    "app_feature_2": {"display_name": "Ancienneté Emploi (années)", "min_val": 0.0, "max_val": 50.0, "type": "numerical"},
    "app_feature_3": {"display_name": "Source Extérieure 1", "min_val": 0.0, "max_val": 1.0, "type": "numerical"},
    "app_feature_4": {"display_name": "Source Extérieure 2", "min_val": 0.0, "max_val": 1.0, "type": "numerical"},
    "app_feature_5": {"display_name": "Source Extérieure 3", "min_val": 0.0, "max_val": 1.0, "type": "numerical"},
    "app_feature_6": {"display_name": "Nombre Enfants", "min_val": 0.0, "max_val": 10.0, "type": "numerical"},
    "app_feature_7": {"display_name": "Montant Annuité", "min_val": 1000.0, "max_val": 100000.0, "type": "numerical"},
    "app_feature_8": {"display_name": "Age Client (années)", "min_val": 18.0, "max_val": 70.0, "type": "numerical"},
    # Bureau features (simulées)
    "bureau_debt_ratio": {"display_name": "Ratio Dette Bureau", "min_val": 0.0, "max_val": 5.0, "type": "numerical"},
    "bureau_active_loans_perc": {"display_name": "% Prêts Bureau Actifs", "min_val": 0.0, "max_val": 1.0, "type": "numerical"},
    # Previous Application features (simulées)
    "prev_app_credit_ratio": {"display_name": "Ratio Crédit Préc. App.", "min_val": 0.0, "max_val": 1.0, "type": "numerical"},
    "prev_app_refused_perc": {"display_name": "% App. Préc. Refusées", "min_val": 0.0, "max_val": 1.0, "type": "numerical"},
    # Exemples de features catégorielles si vous voulez les gérer directement dans SHAP_IMPORTANT_FEATURES_INFO
    # "NAME_CONTRACT_TYPE": {"display_name": "Type de Contrat", "options": ["Cash", "Revolving"], "type": "categorical"},
    # "CODE_GENDER": {"display_name": "Genre", "options": ["M", "F", "XNA"], "type": "categorical"},
}

# --- Fonctions Stubs pour l'Ingénierie des Caractéristiques (CONSERVÉES) ---
# Ces fonctions génèrent des données aléatoires ou des noms de colonnes génériques.
# Elles DOIVENT être exactement les mêmes que celles que vous simulerez dans Streamlit.

def load_application_data_stub(debug_mode=False):
    """Simule le chargement des données d'application et la création de features initiales."""
    num_rows = 3000 if debug_mode else 30000
    
    data = {}
    # Générer des features numériques basées sur SHAP_IMPORTANT_FEATURES_INFO
    for feature_name, info in SHAP_IMPORTANT_FEATURES_INFO.items():
        if info["type"] == "numerical" and feature_name.startswith('app_feature_'):
            min_v, max_v = info["min_val"], info["max_val"]
            data[feature_name] = np.random.rand(num_rows) * (max_v - min_v) + min_v
    
    # Ajouter des features numériques génériques supplémentaires si nécessaire pour le pipeline
    for i in range(len(SHAP_IMPORTANT_FEATURES_INFO), 50): # S'assurer d'avoir au moins 50 app_features
        feat_name = f'app_feature_{i}'
        if feat_name not in data:
            data[feat_name] = np.random.rand(num_rows) * 100 

    df = pd.DataFrame(data)
    df['TARGET'] = np.random.randint(0, 2, num_rows)
    df['SK_ID_CURR'] = np.arange(num_rows)

    # Ajouter des colonnes catégorielles pour simuler
    df['NAME_CONTRACT_TYPE'] = np.random.choice(['Cash', 'Revolving'], num_rows)
    df['CODE_GENDER'] = np.random.choice(['M', 'F', 'XNA'], num_rows)
    df['FLAG_OWN_CAR'] = np.random.choice(['Y', 'N'], num_rows)
    df['NAME_INCOME_TYPE'] = np.random.choice(['Working', 'Commercial associate', 'Pensioner', 'State servant'], num_rows)
    
    # Ajouter une colonne avec quelques NaN pour tester l'imputation
    if 'app_feature_0' in df.columns:
        df['app_feature_0'].iloc[::100] = np.nan 
    else:
        df['another_feature_with_nan'] = np.random.rand(num_rows)
        df['another_feature_with_nan'].iloc[::100] = np.nan

    logger.info(f"Initial DataFrame shape from application_train_test (stub): {df.shape}")
    return df

def process_bureau_data_stub(df_app_main, debug_mode=False):
    """Simule l'ajout de features agrégées de bureau."""
    # Créer et joindre les features bureau simulées, en utilisant SHAP_IMPORTANT_FEATURES_INFO
    if 'bureau_debt_ratio' not in df_app_main.columns:
        info = SHAP_IMPORTANT_FEATURES_INFO.get('bureau_debt_ratio', {"min_val": 0.0, "max_val": 5.0})
        df_app_main['bureau_debt_ratio'] = np.random.rand(len(df_app_main)) * (info["max_val"] - info["min_val"]) + info["min_val"]
    
    if 'bureau_active_loans_perc' not in df_app_main.columns:
        info = SHAP_IMPORTANT_FEATURES_INFO.get('bureau_active_loans_perc', {"min_val": 0.0, "max_val": 1.0})
        df_app_main['bureau_active_loans_perc'] = np.random.rand(len(df_app_main)) * (info["max_val"] - info["min_val"]) + info["min_val"]

    # Ajouter d'autres features bureau génériques si nécessaire
    for i in range(5):
        if f'bureau_feat_gen_{i}' not in df_app_main.columns:
            df_app_main[f'bureau_feat_gen_{i}'] = np.random.rand(len(df_app_main)) * 10
    logger.info("[Process bureau and bureau_balance] done (stub)")
    return df_app_main

def process_previous_applications_data_stub(df_app_main, debug_mode=False):
    """Simule l'ajout de features agrégées de précédentes applications."""
    if 'prev_app_credit_ratio' not in df_app_main.columns:
        info = SHAP_IMPORTANT_FEATURES_INFO.get('prev_app_credit_ratio', {"min_val": 0.0, "max_val": 1.0})
        df_app_main['prev_app_credit_ratio'] = np.random.rand(len(df_app_main)) * (info["max_val"] - info["min_val"]) + info["min_val"]
    
    if 'prev_app_refused_perc' not in df_app_main.columns:
        info = SHAP_IMPORTANT_FEATURES_INFO.get('prev_app_refused_perc', {"min_val": 0.0, "max_val": 1.0})
        df_app_main['prev_app_refused_perc'] = np.random.rand(len(df_app_main)) * (info["max_val"] - info["min_val"]) + info["min_val"]

    for i in range(5):
        if f'prev_app_feat_gen_{i}' not in df_app_main.columns:
            df_app_main[f'prev_app_feat_gen_{i}'] = np.random.rand(len(df_app_main)) * 5
    logger.info("[Process previous_applications] done (stub)")
    return df_app_main

# Les fonctions suivantes peuvent être similaires ou plus simples si elles ne créent pas de features SHAP importantes spécifiques.
def process_pos_cash_data_stub(df_app_main, debug_mode=False):
    for i in range(3):
        if f'pos_feat_gen_{i}' not in df_app_main.columns:
            df_app_main[f'pos_feat_gen_{i}'] = np.random.rand(len(df_app_main)) * 10
    logger.info("[Process POS-CASH balance] done (stub)")
    return df_app_main

def process_installments_payments_data_stub(df_app_main, debug_mode=False):
    for i in range(3):
        if f'install_feat_gen_{i}' not in df_app_main.columns:
            df_app_main[f'install_feat_gen_{i}'] = np.random.rand(len(df_app_main)) * 20
    logger.info("[Process installments payments] done (stub)")
    return df_app_main

def process_credit_card_balance_data_stub(df_app_main, debug_mode=False):
    for i in range(3):
        if f'cc_feat_gen_{i}' not in df_app_main.columns:
            df_app_main[f'cc_feat_gen_{i}'] = np.random.rand(len(df_app_main)) * 50
    logger.info("[Process credit card balance] done (stub)")
    return df_app_main

def run_feature_engineering_pipeline(debug_mode=False):
    """Exécute la chaîne de fonctions d'ingénierie des caractéristiques (stubs)."""
    logger.info("## Étape 1: Ingénierie des Caractéristiques (Stubs)")
    df = load_application_data_stub(debug_mode)
    df = process_bureau_data_stub(df, debug_mode)
    df = process_previous_applications_data_stub(df, debug_mode)
    df = process_pos_cash_data_stub(df, debug_mode)
    df = process_installments_payments_data_stub(df, debug_mode)
    df = process_credit_card_balance_data_stub(df, debug_mode)
    logger.info(f"[Feature Engineering Pipeline Stubs] done. Final DataFrame shape: {df.shape}")
    
    # Nettoyage mémoire
    gc.collect() 
    return df

# --- Fonction pour trouver le meilleur seuil de classification ---
def find_best_threshold_and_fbeta(y_true, y_probs, beta):
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        logger.warning("y_true est vide ou ne contient qu'une seule classe. Impossible de calculer le seuil optimal. Retourne 0.5.")
        return 0.5, 0.0

    thresholds = np.linspace(0, 1, 1000)
    best_fbeta = -1
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        if len(np.unique(y_pred)) < 2: # Évite les erreurs si une prédiction est uniforme
            continue
        current_fbeta = fbeta_score(y_true, y_pred, beta=beta, average='binary', pos_label=1)
        if current_fbeta > best_fbeta:
            best_fbeta = current_fbeta
            best_threshold = threshold
    return best_threshold, best_fbeta

# --- Fonction Principale du Pipeline de Régression Logistique ---
def main_logistic_regression_pipeline(debug_mode=True, register_model=True):
    """
    Exécute le pipeline complet d'entraînement du modèle de régression logistique.
    Args:
        debug_mode (bool): Si True, utilise un petit sous-ensemble de données pour un test rapide.
        register_model (bool): Si True, enregistre le modèle entraîné dans MLflow Model Registry.
    """
    logger.info("Début du pipeline de Régression Logistique (avec Stubs).")

    with mlflow.start_run(run_name="Logistic_Regression_Training_Stubs") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        logger.info(f"MLflow Artifact URI for this run: {mlflow.get_artifact_uri()}")

        # 1. Ingénierie des Caractéristiques (via les Stubs)
        full_df = run_feature_engineering_pipeline(debug_mode)

        X = full_df.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore') # Ignore SK_ID_CURR si non présent
        y = full_df['TARGET']

        # Identifier les colonnes numériques et catégorielles PRÉSENTES dans X
        # Assurez-vous que les types sont corrects après les stubs
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = X[col].astype('category')
            elif pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
        X.replace([np.inf, -np.inf], np.nan, inplace=True) # Remplacer inf par NaN pour l'imputation

        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include='category').columns.tolist()
        
        # Filtrer et valider: S'assurer que les features SHAP importantes sont numériques
        validated_shap_features_info = {}
        for feat_name, info in SHAP_IMPORTANT_FEATURES_INFO.items():
            if info["type"] == "numerical" and feat_name in numerical_features:
                validated_shap_features_info[feat_name] = info
            elif info["type"] == "categorical" and feat_name in categorical_features:
                 validated_shap_features_info[feat_name] = info # Inclure les cat si on veut les slider plus tard
            else:
                logger.warning(f"La feature importante SHAP '{feat_name}' n'est pas du type attendu ou n'a pas été trouvée dans les données après l'ingénierie des caractéristiques. Elle sera ignorée pour Streamlit.")
        
        features_for_streamlit = validated_shap_features_info.copy() 
        
        # Fallback robuste: Si features_for_streamlit est vide, utiliser toutes les features numériques
        if not features_for_streamlit and numerical_features:
            logger.warning("Aucune des features SHAP importantes spécifiées n'a été validée ou n'a été trouvée. "
<<<<<<< HEAD
                             "Toutes les features numériques détectées seront utilisées pour Streamlit comme fallback.")
=======
                            "Toutes les features numériques détectées seront utilisées pour Streamlit comme fallback.")
>>>>>>> f19f2d7b0059efff3b5e238d74411ad289f4909f
            for feat_name in numerical_features:
                min_val = float(X[feat_name].min()) if not X[feat_name].empty and not pd.isna(X[feat_name].min()) else 0.0
                max_val = float(X[feat_name].max()) if not X[feat_name].empty and not pd.isna(X[feat_name].max()) else 1.0
                features_for_streamlit[feat_name] = {
                    "display_name": feat_name,
                    "min_val": min_val,
                    "max_val": max_val,
                    "type": "numerical"
                }

        # Liste de toutes les features sur lesquelles le modèle sera entraîné
        all_training_features = numerical_features + categorical_features
        
        logger.info("\n--- Variables (features) utilisées pour l'entraînement du modèle (passées au ColumnTransformer) ---")
        logger.info(f"Nombre de variables : {len(all_training_features)}")
        logger.info("Liste des variables :\n - " + "\n - ".join(all_training_features))
        
        logger.info("\n--- Variables (features) importantes logguées pour Streamlit ---")
        logger.info(f"Nombre de variables : {len(features_for_streamlit)}")
        for feat_name, info in features_for_streamlit.items():
            logger.info(f" - {feat_name} (Display: {info['display_name']}, Type: {info['type']}, Range: [{info.get('min_val', 'N/A'):.2f}-{info.get('max_val', 'N/A'):.2f}])")
        logger.info("--------------------------------------------------------------------------------\n")

        # 2. Séparation des Données - Utilise TOUTES les features que le pipeline est configuré pour traiter
        X_train, X_test, y_train, y_test = train_test_split(
            X[all_training_features], y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Training data shape before preprocessing (for pipeline): {X_train.shape}")
        logger.info(f"Test data shape before preprocessing (for pipeline): {X_test.shape}")

        # 3. Définition du Pipeline Complet (Préprocesseur + Modèle)
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )

        logistic_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver='liblinear', random_state=42, n_jobs=-1, max_iter=1000))
        ])
        logger.info("## Étape 3: Définition du Pipeline Complet (Préprocesseur + Modèle)")

        # 4. Optimisation des Hyperparamètres avec GridSearchCV
        logger.info("## Étape 4: Optimisation des Hyperparamètres avec GridSearchCV sur le Pipeline Complet")
        param_grid = {
            'classifier__C': [0.0001, 0.001, 0.01, 0.1, 1, 10]
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            logistic_pipeline,
            param_grid,
            cv=cv,
            scoring=fbeta_scorer,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score_cv = grid_search.best_score_

        logger.info("\n--- Résultats de GridSearchCV ---")
        logger.info(f"Meilleurs hyperparamètres: {best_params}")
        logger.info(f"Meilleur score F-beta ({CUSTOM_FBETA_BETA}) sur CV: {best_score_cv:.4f}")

        # Collecte des probabilités Out-Of-Fold (OOF) pour trouver le seuil optimal
        logger.info("\n--- Collecte des probabilités Out-Of-Fold avec le meilleur modèle ---")
        oof_probs = np.zeros(len(X_train))
        oof_y_true = y_train.copy()

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            fold_X_train, fold_X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            fold_y_train, fold_y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            fold_pipeline = logistic_pipeline.set_params(**best_params)
            fold_pipeline.fit(fold_X_train, fold_y_train)

            oof_probs[val_idx] = fold_pipeline.predict_proba(fold_X_val)[:, 1]

        optimal_fbeta_threshold_oof_mean, oof_fbeta_score_at_optimal_threshold = \
            find_best_threshold_and_fbeta(oof_y_true, oof_probs, CUSTOM_FBETA_BETA)
            
        logger.info(f"Seuil optimal F-beta ({CUSTOM_FBETA_BETA}) moyen sur les probabilités OOF: {optimal_fbeta_threshold_oof_mean:.4f}")
        logger.info(f"Score F-beta ({CUSTOM_FBETA_BETA}) OOF avec seuil optimal moyen: {oof_fbeta_score_at_optimal_threshold:.4f}")
        logger.info("[GridSearchCV for Logistic Regression Pipeline] done")


        # 5. Entraînement du Modèle Final sur l'Ensemble d'Entraînement Complet et Évaluation
        logger.info("\n## Étape 5: Entraînement du Modèle Final (Pipeline Complet) et Prédiction sur le Jeu de Test")
        logger.info("--- Training final full pipeline on full training data using best hyperparameters ---")
            
        final_pipeline_model = logistic_pipeline.set_params(**best_params)
        final_pipeline_model.fit(X_train, y_train)

        y_train_pred_proba = final_pipeline_model.predict_proba(X_train)[:, 1]
        y_train_pred_optimal_threshold = (y_train_pred_proba >= optimal_fbeta_threshold_oof_mean).astype(int)

        logger.info("\n--- Métriques du modèle final sur l'ensemble d'entraînement complet (avec seuil optimal) ---")
        metrics = {
            "Accuracy": accuracy_score(y_train, y_train_pred_optimal_threshold),
            "Recall": recall_score(y_train, y_train_pred_optimal_threshold),
            "Precision": precision_score(y_train, y_train_pred_optimal_threshold, zero_division=0),
            "F1-Score": f1_score(y_train, y_train_pred_optimal_threshold),
            f"F{CUSTOM_FBETA_BETA}-Score": fbeta_score(y_train, y_train_pred_optimal_threshold, beta=CUSTOM_FBETA_BETA, average='binary', pos_label=1),
            "AUC Score": roc_auc_score(y_train, y_train_pred_proba)
        }

        logger.info(f" Final_train_set Metrics (Threshold={optimal_fbeta_threshold_oof_mean:.4f}):")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
            mlflow.log_metric(f"train_{metric_name.replace('-', '_').lower()}", value)

        # Log des paramètres et du modèle avec MLflow
        mlflow.log_params(best_params)
        mlflow.log_param("optimal_threshold_value", optimal_fbeta_threshold_oof_mean)
        
        # Log de toutes les features utilisées pour l'entraînement (pour que Streamlit puisse reconstruire le DF complet)
        mlflow.log_param("all_training_features_names", json.dumps(all_training_features))

        # Enregistrement des informations sur les features importantes pour Streamlit
        mlflow.log_param("features_info_for_streamlit_json", json.dumps(features_for_streamlit))

        # Création de la signature du modèle pour MLflow
        # Utilise X_train.head(5) pour infer_signature, il est crucial que ce DataFrame
        # ait les mêmes colonnes et types que les données d'entraînement.
        model_signature = infer_signature(
            X_train.head(5), 
            final_pipeline_model.predict_proba(X_train.head(5))
        )
        
        # Enregistrement du modèle dans le MLflow Model Registry (vers S3)
        if register_model:
            logger.info(f"Attempting to register model '{MLFLOW_MODEL_NAME}' in MLflow Model Registry...")
            try:
                mlflow.sklearn.log_model(
                    sk_model=final_pipeline_model,
                    artifact_path="logistic_regression_pipeline", # Chemin de l'artefact dans le run
                    registered_model_name=MLFLOW_MODEL_NAME,
                    signature=model_signature,
                    input_example=X_train.head(1), # Fournit un exemple d'entrée (après FE stub)
                    metadata={ # Ces métadonnées apparaissent dans les tags du modèle dans MLflow UI
                        "optimal_threshold": str(optimal_fbeta_threshold_oof_mean),
                        "features_info_for_streamlit": json.dumps(features_for_streamlit), 
                        "all_training_features": json.dumps(all_training_features) 
                    }
                )
                logger.info(f"Modèle '{MLFLOW_MODEL_NAME}' enregistré avec succès dans le MLflow Model Registry (vers S3).")
            except Exception as e:
                logger.error(f"Échec de l'enregistrement du modèle dans MLflow Model Registry: {e}")
                logger.error("Vérifiez les permissions S3, la configuration du registre ou le nom du modèle.")
        else:
            logger.info("Modèle non enregistré dans le Model Registry (register_model=False).")

        logger.info(f"MLflow Run completed. View at {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

# --- Exécution du Script d'Entraînement ---
if __name__ == "__main__":
    main_logistic_regression_pipeline(debug_mode=True, register_model=True)