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

# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration MLflow ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
except Exception as e:
    logger.error(f"Failed to set MLflow tracking URI: {e}")
    logger.error("MLflow tracking might not work correctly. Ensure MLflow UI is running and accessible.")
    # Quitter si la configuration MLflow échoue, car c'est critique
    exit(1)

# IMPORTANT: Si vous avez eu l'erreur "Cannot set a deleted experiment",
# vous pouvez changer le nom de l'expérience ici pour en créer une nouvelle.
# Ou assurez-vous d'avoir purgé l'ancienne via MLflow UI ou CLI (voir nos discussions précédentes).
MLFLOW_EXPERIMENT_NAME = "Home_Credit_Default_Risk_Logistic_Regression_Pipeline_Final_Deployment" # Nom mis à jour
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

MLFLOW_MODEL_NAME = "HomeCreditLogisticRegressionPipeline"

CUSTOM_FBETA_BETA = np.sqrt(10)
fbeta_scorer = make_scorer(fbeta_score, beta=CUSTOM_FBETA_BETA, average='binary', pos_label=1)
logger.info(f"Le seuil de classification sera optimisé pour le F-beta score (beta={CUSTOM_FBETA_BETA}).")

# --- PLACEHOLDER : Mapping des noms de features SHAP importants vers des noms explicites ---
# VOUS DEVEZ ABSOLUMENT ADAPTER CE DICTIONNAIRE AVEC VOS VRAIS NOMS DE FEATURES ET LEURS INTERVALLES RÉELS
# basés sur votre analyse SHAP et vos données d'entraînement réelles.
# Si vous remplacez les fonctions stubs par votre code réel de feature engineering, ces noms DOIVENT correspondre
# aux noms de colonnes résultants dans votre DataFrame.
SHAP_IMPORTANT_FEATURES_INFO = {
    # Ces noms 'app_feature_X' doivent correspondre aux noms générés par vos fonctions stub
    # ou aux VRAIS NOMS de colonnes de votre jeu de données après feature engineering.
    # Exemple si vous aviez des features réelles comme 'ANNUITY_INCOME_RATIO'
    "app_feature_15": {"display_name": "Ratio Crédit/Annuité", "min_val": 0.01, "max_val": 10.0},
    "app_feature_33": {"display_name": "Ancienneté Emploi (années)", "min_val": 0.0, "max_val": 50.0},
    "app_feature_21": {"display_name": "Taux Population Région", "min_val": 0.001, "max_val": 0.1},
    "app_feature_19": {"display_name": "Source Extérieure 1", "min_val": 0.0, "max_val": 1.0},
    "app_feature_31": {"display_name": "Source Extérieure 2", "min_val": 0.0, "max_val": 1.0},
    "app_feature_24": {"display_name": "Source Extérieure 3", "min_val": 0.0, "max_val": 1.0},
    "app_feature_45": {"display_name": "Nombre Enfants", "min_val": 0.0, "max_val": 10.0},
    "app_feature_16": {"display_name": "Montant Annuité", "min_val": 1000.0, "max_val": 100000.0},
    "app_feature_17": {"display_name": "Age Client (années)", "min_val": 18.0, "max_val": 70.0},
    # Ajoutez d'autres features SHAP importantes si nécessaire, assurez-vous qu'elles sont générées dans les stubs
    # exemple: "BUREAU_TOTAL_DEBT": {"display_name": "Dette Totale Bureau", "min_val": 0.0, "max_val": 500000.0}
}
# La liste des noms de features SHAP importantes que nous voulons voir dans Streamlit
SHAP_IMPORTANT_FEATURES_NAMES = list(SHAP_IMPORTANT_FEATURES_INFO.keys())


# --- Fonctions Stubs pour l'Ingénierie des Caractéristiques (À REMPLACER) ---
# IMPORTANT: CES FONCTIONS GÉNÈRENT DES DONNÉES ALÉATOIRES POUR SIMPLIFIER.
# VOUS DEVEZ REMPLACER CES FONCTIONS PAR VOTRE VRAIE LOGIQUE DE FEATURE ENGINEERING
# pour que les noms des colonnes correspondent à vos vraies features et à SHAP_IMPORTANT_FEATURES_INFO.

def load_application_data_stub(debug_mode=False):
    """Simule le chargement des données d'application et la création de features initiales."""
    num_rows = 3000 if debug_mode else 30000 # Réduit le nombre de lignes en debug mode
    
    data = {}
    # Générer des features numériques de 'app_feature_0' à 'app_feature_49'
    for i in range(50):
        feature_name = f'app_feature_{i}'
        info = SHAP_IMPORTANT_FEATURES_INFO.get(feature_name, {"min_val": 0.0, "max_val": 1.0})
        min_v, max_v = info["min_val"], info["max_val"]
        data[feature_name] = np.random.rand(num_rows) * (max_v - min_v) + min_v
    
    # S'assurer que les features SHAP importantes qui ne sont PAS des app_feature_X sont générées
    # (par exemple, si vous aviez une feature 'BUREAU_TOTAL_DEBT')
    for feature in SHAP_IMPORTANT_FEATURES_NAMES:
        if not feature.startswith('app_feature_') and feature not in data:
            info = SHAP_IMPORTANT_FEATURES_INFO.get(feature, {"min_val": 0.0, "max_val": 1.0})
            min_v, max_v = info["min_val"], info["max_val"]
            data[feature] = np.random.rand(num_rows) * (max_v - min_v) + min_v

    df = pd.DataFrame(data)
    df['TARGET'] = np.random.randint(0, 2, num_rows)
    df['SK_ID_CURR'] = np.arange(num_rows) # ID client

    # Ajouter des colonnes catégorielles pour simuler
    df['NAME_CONTRACT_TYPE'] = np.random.choice(['Cash', 'Revolving'], num_rows)
    df['CODE_GENDER'] = np.random.choice(['M', 'F', 'XNA'], num_rows)
    df['FLAG_OWN_CAR'] = np.random.choice(['Y', 'N'], num_rows)
    df['NAME_INCOME_TYPE'] = np.random.choice(['Working', 'Commercial associate', 'Pensioner', 'State servant'], num_rows)
    # Ajouter une colonne avec quelques NaN pour tester l'imputation
    df['app_feature_0'].iloc[::100] = np.nan 

    logger.info(f"Initial DataFrame shape from application_train_test (stub): {df.shape}")
    return df

# Ces fonctions simulent l'ajout de features depuis d'autres sources.
# Elles doivent être remplacées par votre code réel.
def process_bureau_data_stub(df, debug_mode=False):
    if 'BUREAU_TOTAL_DEBT' in SHAP_IMPORTANT_FEATURES_INFO and 'BUREAU_TOTAL_DEBT' not in df.columns:
        info = SHAP_IMPORTANT_FEATURES_INFO['BUREAU_TOTAL_DEBT']
        min_v, max_v = info["min_val"], info["max_val"]
        df['BUREAU_TOTAL_DEBT'] = np.random.rand(len(df)) * (max_v - min_v) + min_v
    for i in range(5): # Ajoute d'autres features 'bureau' génériques
        if f'bureau_feat_{i}' not in df.columns:
            df[f'bureau_feat_{i}'] = np.random.rand(len(df))
    logger.info("[Process bureau and bureau_balance] done (stub)")
    return df

def process_previous_applications_data_stub(df, debug_mode=False):
    for i in range(5):
        if f'prev_app_feat_{i}' not in df.columns:
            df[f'prev_app_feat_{i}'] = np.random.rand(len(df))
    logger.info("[Process previous_applications] done (stub)")
    return df

def process_pos_cash_data_stub(df, debug_mode=False):
    for i in range(5):
        if f'pos_feat_{i}' not in df.columns:
            df[f'pos_feat_{i}'] = np.random.rand(len(df))
    logger.info("[Process POS-CASH balance] done (stub)")
    return df

def process_installments_payments_data_stub(df, debug_mode=False):
    for i in range(5):
        if f'install_feat_{i}' not in df.columns:
            df[f'install_feat_{i}'] = np.random.rand(len(df))
    logger.info("[Process installments payments] done (stub)")
    return df

def process_credit_card_balance_data_stub(df, debug_mode=False):
    for i in range(5):
        if f'cc_feat_{i}' not in df.columns:
            df[f'cc_feat_{i}'] = np.random.rand(len(df))
    logger.info("[Process credit card balance] done (stub)")
    return df

def run_feature_engineering_pipeline(debug_mode=False):
    """Exécute la chaîne de fonctions d'ingénierie des caractéristiques (stubs)."""
    logger.info("## Étape 1: Ingénierie des Caractéristiques")
    df = load_application_data_stub(debug_mode)
    df = process_bureau_data_stub(df, debug_mode)
    df = process_previous_applications_data_stub(df, debug_mode)
    df = process_pos_cash_data_stub(df, debug_mode)
    df = process_installments_payments_data_stub(df, debug_mode)
    df = process_credit_card_balance_data_stub(df, debug_mode)
    logger.info(f"[Feature Engineering Pipeline] done. Final DataFrame shape after all joins: {df.shape}")
    return df

# --- Fonction pour trouver le meilleur seuil de classification ---
def find_best_threshold_and_fbeta(y_true, y_probs, beta):
    """
    Calcule le seuil optimal pour maximiser le F-beta score
    sur un ensemble de probabilités prédites.
    """
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        logger.warning("y_true est vide ou ne contient qu'une seule classe. Impossible de calculer le seuil optimal. Retourne 0.5.")
        return 0.5, 0.0 # Retourne un seuil par défaut et un score de 0.0

    thresholds = np.linspace(0, 1, 1000)
    best_fbeta = -1
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        # S'assurer que les deux classes sont présentes après binarisation pour fbeta_score
        if len(np.unique(y_pred)) < 2:
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
    logger.info("Début du pipeline de Régression Logistique.")

    with mlflow.start_run(run_name="Logistic_Regression_Training_Run") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        logger.info(f"MLflow Artifact URI for this run: {mlflow.get_artifact_uri()}")

        # 1. Ingénierie des Caractéristiques
        full_df = run_feature_engineering_pipeline(debug_mode)

        X = full_df.drop(columns=['TARGET', 'SK_ID_CURR'])
        y = full_df['TARGET']

        # Identifier les colonnes numériques et catégorielles PRÉSENTES dans X
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include='object').columns.tolist()
        
        # Filtrer et valider: S'assurer que les features SHAP importantes sont numériques
        # et présentes dans le DataFrame final (après FE).
        validated_shap_features_info = {}
        for feat_name, info in SHAP_IMPORTANT_FEATURES_INFO.items():
            if feat_name in numerical_features: # Ne considérer que les features numériques pour les sliders de Streamlit
                validated_shap_features_info[feat_name] = info
            else:
                logger.warning(f"La feature importante SHAP '{feat_name}' n'est pas numérique ou n'a pas été trouvée dans les données après l'ingénierie des caractéristiques. Elle sera ignorée pour Streamlit.")
        
        # Copie profonde pour éviter des modifications inattendues
        features_for_streamlit = validated_shap_features_info.copy() 
        
        # Fallback robuste: Si features_for_streamlit est vide (aucune feature SHAP valide n'a été trouvée),
        # utilisez toutes les features numériques comme fallback pour Streamlit.
        if not features_for_streamlit:
            logger.warning("Aucune des features SHAP importantes spécifiées n'a été validée ou n'a été trouvée. "
                           "Toutes les features numériques détectées seront utilisées pour Streamlit comme fallback.")
            for feat_name in numerical_features:
                # Utilisez min/max de X pour la plage, même si ce ne sont pas les valeurs réelles d'entraînement
                # C'est un fallback acceptable si SHAP_IMPORTANT_FEATURES_INFO est vide ou incorrect.
                min_val = float(X[feat_name].min()) if not X[feat_name].empty else 0.0
                max_val = float(X[feat_name].max()) if not X[feat_name].empty else 1.0
                features_for_streamlit[feat_name] = {
                    "display_name": feat_name,
                    "min_val": min_val,
                    "max_val": max_val
                }

        # Liste de toutes les features sur lesquelles le modèle sera entraîné
        # Cela garantit que le ColumnTransformer utilise les colonnes qui existent réellement après FE
        all_training_features = numerical_features + categorical_features
        
        logger.info(f"DEBUG: features_for_streamlit content BEFORE JSON dump: {features_for_streamlit}")

        logger.info("\n--- Variables (features) utilisées pour l'entraînement du modèle (passées au ColumnTransformer) ---")
        logger.info(f"Nombre de variables : {len(all_training_features)}")
        logger.info("Liste des variables :\n - " + "\n - ".join(all_training_features))
        
        logger.info("\n--- Variables (features) importantes logguées pour Streamlit ---")
        logger.info(f"Nombre de variables : {len(features_for_streamlit)}")
        for feat_name, info in features_for_streamlit.items():
            logger.info(f" - {feat_name} (Display: {info['display_name']}, Range: [{info['min_val']:.2f}-{info['max_val']:.2f}])")
        logger.info("--------------------------------------------------------------------------------\n")

        # 2. Séparation des Données - Utilise TOUTES les features que le pipeline est configuré pour traiter
        # Important: Ne pas inclure TARGET/SK_ID_CURR
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
            remainder='passthrough' # Conserver les colonnes non spécifiées (si existantes, mais ici elles ne devraient pas l'être)
        )

        logistic_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver='liblinear', random_state=42, n_jobs=-1, max_iter=1000)) # Ajout de max_iter
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
        
        # Enregistrement du modèle dans le MLflow Model Registry
        if register_model:
            logger.info(f"Attempting to register model '{MLFLOW_MODEL_NAME}' in MLflow Model Registry...")
            try:
                mlflow.sklearn.log_model(
                    sk_model=final_pipeline_model,
                    artifact_path="logistic_regression_model",
                    registered_model_name=MLFLOW_MODEL_NAME,
                    signature=model_signature,
                    input_example=X_train.head(1), # Fournit un exemple d'entrée
                    metadata={ # Ces métadonnées apparaissent dans les tags du modèle dans MLflow UI
                        "optimal_threshold": str(optimal_fbeta_threshold_oof_mean),
                        "features_info_for_streamlit": json.dumps(features_for_streamlit), 
                        "all_training_features": json.dumps(all_training_features) 
                    }
                )
                logger.info(f"Modèle '{MLFLOW_MODEL_NAME}' enregistré avec succès dans le MLflow Model Registry.")
            except Exception as e:
                logger.error(f"Échec de l'enregistrement du modèle dans MLflow Model Registry: {e}")
                logger.error("Vérifiez les permissions, la configuration du registre ou le nom du modèle.")
        else:
            logger.info("Modèle non enregistré dans le Model Registry (register_model=False).")

        logger.info(f"MLflow Run completed. View at {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

# --- Exécution du Script d'Entraînement ---
if __name__ == "__main__":
    main_logistic_regression_pipeline(debug_mode=True, register_model=True)