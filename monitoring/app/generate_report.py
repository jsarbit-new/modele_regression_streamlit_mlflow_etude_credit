import pandas as pd
import numpy as np
import os
import gc

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.model_profile import ColumnMapping

# --- Chemins de fichiers dans le conteneur Docker ---
# C'est le point de montage de votre dossier C:\Users\jonjo\Documents\open classrooms\Projet 7\input
INPUT_DATA_DIR = '/app/data_input/'

# --- Fonctions de Feature Engineering (COPIÉES DE VOTRE SCRIPT D'ENTRAÎNEMENT) ---
# Assurez-vous que cette fonction est bien définie ici
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtypes == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Fonction principale pour charger et prétraiter les données
def load_and_preprocess_data_for_evidently(num_rows=None, nan_as_category=False):
    print(f"Tentative de lecture de : {os.path.join(INPUT_DATA_DIR, 'application_train.csv')}")
    # Charge les fichiers bruts depuis le dossier monté
    # RENOMMÉES EN application_train.csv et application_test.csv
    with_target_df = pd.read_csv(os.path.join(INPUT_DATA_DIR, 'application_train.csv'), nrows=num_rows)
    without_target_df = pd.read_csv(os.path.join(INPUT_DATA_DIR, 'application_test.csv'), nrows=num_rows)

    # Stocke les IDs et la cible pour la séparation après FE
    train_ids = with_target_df['SK_ID_CURR']
    train_target = with_target_df['TARGET']
    test_ids = without_target_df['SK_ID_CURR']

    # Concaténez les DataFrames pour appliquer le FE commun (sans la colonne TARGET pour éviter les problèmes)
    if 'TARGET' in with_target_df.columns:
        full_df = pd.concat([with_target_df.drop(columns=['TARGET']), without_target_df], ignore_index=True)
    else:
        full_df = pd.concat([with_target_df, without_target_df], ignore_index=True)

    full_df = full_df.reset_index(drop=True)
    del with_target_df, without_target_df
    gc.collect()

    # --- Votre logique de Feature Engineering copiée ici ---
    full_df = full_df[full_df['CODE_GENDER'] != 'XNA']
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        full_df[bin_feature], uniques = pd.factorize(full_df[bin_feature])
    full_df, _ = one_hot_encoder(full_df, nan_as_category)

    full_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    full_df['DAYS_EMPLOYED_PERC'] = full_df['DAYS_EMPLOYED'] / full_df['DAYS_BIRTH']
    full_df['INCOME_CREDIT_PERC'] = np.where(full_df['AMT_CREDIT'] == 0, np.nan, full_df['AMT_INCOME_TOTAL'] / full_df['AMT_CREDIT'])
    full_df['INCOME_PER_PERSON'] = np.where(full_df['CNT_FAM_MEMBERS'] == 0, np.nan, full_df['AMT_INCOME_TOTAL'] / full_df['CNT_FAM_MEMBERS'])
    full_df['ANNUITY_INCOME_PERC'] = np.where(full_df['AMT_INCOME_TOTAL'] == 0, np.nan, full_df['AMT_ANNUITY'] / full_df['AMT_INCOME_TOTAL'])
    full_df['PAYMENT_RATE'] = np.where(full_df['AMT_CREDIT'] == 0, np.nan, full_df['AMT_ANNUITY'] / full_df['AMT_CREDIT'])
    # --- FIN de votre logique de Feature Engineering ---

    # TODO: Ajoutez ici la logique pour joindre les autres fichiers (bureau, previous_applications, etc.)
    # et appliquer leur Feature Engineering sur 'full_df'.
    # EXEMPLE (vous devez avoir ces fonctions de traitement réelles):
    # bureau_df = pd.read_csv(os.path.join(INPUT_DATA_DIR, 'bureau.csv'))
    # bureau_balance_df = pd.read_csv(os.path.join(INPUT_DATA_DIR, 'bureau_balance.csv'))
    # full_df = your_bureau_processing_function(full_df, bureau_df, bureau_balance_df)
    # prev_app_df = pd.read_csv(os.path.join(INPUT_DATA_DIR, 'previous_application.csv'))
    # full_df = your_prev_app_processing_function(full_df, prev_app_df)
    # ... et ainsi de suite pour tous les fichiers CSV bruts que votre modèle utilise.

    # Re-séparer les DataFrames traités
    reference_data_processed = full_df[full_df['SK_ID_CURR'].isin(train_ids)].copy()
    current_data_processed = full_df[full_df['SK_ID_CURR'].isin(test_ids)].copy()

    # Ré-ajouter la colonne TARGET au DataFrame de référence
    if 'TARGET' not in reference_data_processed.columns:
        reference_data_processed = reference_data_processed.merge(train_target.rename('TARGET'),
                                                                  left_on='SK_ID_CURR', right_index=True, how='left')

    return reference_data_processed, current_data_processed

# --- Exécution du chargement et du Feature Engineering ---
print("Début du chargement et de l'ingénierie des caractéristiques pour Evidently...")
try:
    reference_data, current_data = load_and_preprocess_data_for_evidently(num_rows=None, nan_as_category=False)
    print(f"Données de référence traitées (shape): {reference_data.shape}")
    print(f"Données courantes traitées (shape): {current_data.shape}")
except FileNotFoundError as e:
    print(f"Erreur : Un fichier CSV brut est introuvable. Vérifiez les noms et le chemin : {e}")
    print(f"Assurez-vous que tous les fichiers comme application_train.csv, application_test.csv, bureau.csv, etc.")
    print(f"sont présents dans votre dossier C:\\Users\\jonjo\\Documents\\open classrooms\\Projet 7\\input")
    exit(1)


# --- Configuration du mapping des colonnes pour Evidently ---
column_mapping = ColumnMapping()
column_mapping.target = 'TARGET'
column_mapping.prediction = None
column_mapping.id = 'SK_ID_CURR'

# Détectez automatiquement les features numériques et catégorielles après le FE
numerical_features_evidently = reference_data.select_dtypes(include=np.number).columns.tolist()
categorical_features_evidently = reference_data.select_dtypes(include='object').columns.tolist()

# Nettoyage : retirer la cible et l'ID des listes de features à analyser
if column_mapping.target in numerical_features_evidently: numerical_features_evidently.remove(column_mapping.target)
if column_mapping.target in categorical_features_evidently: categorical_features_evidently.remove(column_mapping.target)
if column_mapping.id in numerical_features_evidently: numerical_features_evidently.remove(column_mapping.id)
if column_mapping.id in categorical_features_evidently: categorical_features_evidently.remove(column_mapping.id)


column_mapping.numerical_features = numerical_features_evidently
column_mapping.categorical_features = categorical_features_evidently
column_mapping.datetime = None

# --- Création et exécution du rapport Evidently ---
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

data_drift_report.run(current_data=current_data, reference_data=reference_data, column_mapping=column_mapping)

OUTPUT_REPORT_PATH = 'evidently_data_drift_report.html'
print(f"Génération du rapport Evidently vers {OUTPUT_REPORT_PATH}...")
data_drift_report.save_html(OUTPUT_REPORT_PATH)

print("Rapport généré avec succès.")