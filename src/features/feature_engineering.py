# src/features/feature_engineering.py

import pandas as pd
import numpy as np
import logging
import json
import os

logger = logging.getLogger(__name__)

# --- Placeholder pour les informations sur les features importantes SHAP ---
# NOTE: Ces informations seront chargées DEPUIS S3 par Streamlit en PROD.
# Elles sont ici pour le développement/test du script d'entraînement
# et pour donner une idée de la structure attendue.
SHAP_IMPORTANT_FEATURES_INFO = {
    "app_feature_15": {"display_name": "Ratio Crédit/Annuité", "min_val": 0.01, "max_val": 10.0},
    "app_feature_33": {"display_name": "Ancienneté Emploi (années)", "min_val": 0.0, "max_val": 50.0},
    "app_feature_21": {"display_name": "Taux Population Région", "min_val": 0.001, "max_val": 0.1},
    "app_feature_19": {"display_name": "Source Extérieure 1", "min_val": 0.0, "max_val": 1.0},
    "app_feature_31": {"display_name": "Source Extérieure 2", "min_val": 0.0, "max_val": 1.0},
    "app_feature_24": {"display_name": "Source Extérieure 3", "min_val": 0.0, "max_val": 1.0},
    "app_feature_45": {"display_name": "Nombre Enfants", "min_val": 0.0, "max_val": 10.0},
    "app_feature_16": {"display_name": "Montant Annuité", "min_val": 1000.0, "max_val": 100000.0},
    "app_feature_17": {"display_name": "Age Client (années)", "min_val": 18.0, "max_val": 70.0},
    "BUREAU_TOTAL_DEBT": {"display_name": "Dette Totale Bureau", "min_val": 0.0, "max_val": 500000.0}
}
SHAP_IMPORTANT_FEATURES_NAMES = list(SHAP_IMPORTANT_FEATURES_INFO.keys())


# --- Fonctions Stubs pour l'Ingénierie des Caractéristiques (À REMPLACER si vous avez la vraie logique) ---
# Ces fonctions simulent le chargement et le traitement des données brutes.
# L'idéal est qu'elles reflètent la structure et les types de données de vos VRAIS fichiers CSV.

def load_application_data(debug_mode=False):
    """Simule le chargement des données d'application et la création de features initiales."""
    # --- DÉBUT DU STUB (À REMPLACER PAR VOTRE VRAIE LOGIQUE DE CHARGEMENT ET PRÉPARATION) ---
    num_rows = 3000 if debug_mode else 30000 # Réduit le nombre de lignes en debug mode
    
    data = {}
    # Générer des features numériques de 'app_feature_0' à 'app_feature_49'
    for i in range(50):
        feature_name = f'app_feature_{i}'
        info = SHAP_IMPORTANT_FEATURES_INFO.get(feature_name, {"min_val": 0.0, "max_val": 1.0})
        min_v, max_v = info["min_val"], info["max_val"]
        data[feature_name] = np.random.rand(num_rows) * (max_v - min_v) + min_v
    
    # S'assurer que les features SHAP importantes qui ne sont PAS des app_feature_X sont générées
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
    df.loc[df.sample(frac=0.1).index, 'app_feature_0'] = np.nan # Ajout de NaN aléatoires
    # --- FIN DU STUB ---
    logger.info(f"DataFrame initial from application_train_test (stub/réel): {df.shape}")
    return df

def process_bureau_data(df, debug_mode=False):
    """Implémentez votre VRAIE LOGIQUE de traitement des données `bureau.csv` et `bureau_balance.csv` ici."""
    logger.info("Traitement des données bureau et bureau_balance (remplacez par votre vraie logique)...")
    # --- DÉBUT DU STUB (À REMPLACER) ---
    if 'BUREAU_TOTAL_DEBT' in SHAP_IMPORTANT_FEATURES_INFO and 'BUREAU_TOTAL_DEBT' not in df.columns:
        info = SHAP_IMPORTANT_FEATURES_INFO['BUREAU_TOTAL_DEBT']
        min_v, max_v = info["min_val"], info["max_val"]
        df['BUREAU_TOTAL_DEBT'] = np.random.rand(len(df)) * (max_v - min_v) + min_v
    for i in range(5):
        if f'bureau_feat_{i}' not in df.columns:
            df[f'bureau_feat_{i}'] = np.random.rand(len(df))
    # --- FIN DU STUB ---
    logger.info("[Process bureau and bureau_balance] done.")
    return df

def process_previous_applications_data(df, debug_mode=False):
    """Implémentez votre VRAIE LOGIQUE de traitement des données `previous_application.csv` ici."""
    logger.info("Traitement des données previous_applications (remplacez par votre vraie logique)...")
    # --- DÉBUT DU STUB (À REMPLACER) ---
    for i in range(5):
        if f'prev_app_feat_{i}' not in df.columns:
            df[f'prev_app_feat_{i}'] = np.random.rand(len(df))
    # --- FIN DU STUB ---
    logger.info("[Process previous_applications] done.")
    return df

def process_pos_cash_data(df, debug_mode=False):
    """Implémentez votre VRAIE LOGIQUE de traitement des données `POS_CASH_balance.csv` ici."""
    logger.info("Traitement des données POS-CASH balance (remplacez par votre vraie logique)...")
    # --- DÉBUT DU STUB (À REMPLACER) ---
    for i in range(5):
        if f'pos_feat_{i}' not in df.columns:
            df[f'pos_feat_{i}'] = np.random.rand(len(df))
    # --- FIN DU STUB ---
    logger.info("[Process POS-CASH balance] done.")
    return df

def process_installments_payments_data(df, debug_mode=False):
    """Implémentez votre VRAIE LOGIQUE de traitement des données `installments_payments.csv` ici."""
    logger.info("Traitement des données installments payments (remplacez par votre vraie logique)...")
    # --- DÉBUT DU STUB (À REMPLACER) ---
    for i in range(5):
        if f'install_feat_{i}' not in df.columns:
            df[f'install_feat_{i}'] = np.random.rand(len(df))
    # --- FIN DU STUB ---
    logger.info("[Process installments payments] done.")
    return df

def process_credit_card_balance_data(df, debug_mode=False):
    """Implémentez votre VRAIE LOGIQUE de traitement des données `credit_card_balance.csv` ici."""
    logger.info("Traitement des données credit card balance (remplacez par votre vraie logique)...")
    # --- DÉBUT DU STUB (À REMPLACER) ---
    for i in range(5):
        if f'cc_feat_{i}' not in df.columns:
            df[f'cc_feat_{i}'] = np.random.rand(len(df))
    # --- FIN DU STUB ---
    logger.info("[Process credit card balance] done.")
    return df

def run_all_feature_engineering(debug_mode=False):
    """
    Exécute la chaîne complète des fonctions d'ingénierie des caractéristiques.
    Cette fonction est le point d'entrée pour la création des features.
    """
    logger.info("## Démarrage du pipeline d'Ingénierie des Caractéristiques")
    df = load_application_data(debug_mode)
    df = process_bureau_data(df, debug_mode)
    df = process_previous_applications_data(df, debug_mode)
    df = process_pos_cash_data(df, debug_mode)
    df = process_installments_payments_data(df, debug_mode)
    df = process_credit_card_balance_data(df, debug_mode)
    logger.info(f"[Pipeline d'Ingénierie des Caractéristiques] terminé. Forme finale du DataFrame après toutes les jointures: {df.shape}")
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    processed_data = run_all_feature_engineering(debug_mode=True)
    print("\nFeatures traitées (extrait) :")
    print(processed_data.head())
    print(f"Forme finale : {processed_data.shape}")
    print(f"Colonnes : {processed_data.columns.tolist()}")