import pandas as pd
import numpy as np
import os
import gc

# Imports Evidently (adaptés à la version 0.2.8 de votre conteneur Docker)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
# La classe ColumnMapping n'est pas directement importable via evidently.model_profile
# ni evidently.column_mapping dans la version 0.2.8 de votre conteneur.
# Nous allons utiliser une classe simplifiée et compatible.

# --- Chemins de fichiers dans le conteneur Docker ---
INPUT_DATA_DIR = '/app/data_input/'
OUTPUT_REPORT_PATH = '/app/evidently_data_drift_report.html' # Chemin de sortie du rapport dans le conteneur

# --- Fonctions de Feature Engineering (Copiez et Adaptez les VÔTRES) ---

def one_hot_encoder(df, nan_as_category=True):
    """
    Applique le One-Hot Encoding sur les colonnes catégorielles.
    """
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtypes == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# TODO: Ajoutez ici toutes vos AUTRES fonctions de Feature Engineering si vous en avez.
# Par exemple, des fonctions pour traiter et agréger les données de bureau,
# previous_applications, credit_card_balance, etc., exactement comme dans votre pipeline d'entraînement.


def load_and_preprocess_data_for_evidently(num_rows=None, nan_as_category=False):
    """
    Charge les fichiers CSV bruts, applique le Feature Engineering,
    et retourne les DataFrames traités pour Evidently.
    """
    print(f"Chargement et traitement des données depuis : {INPUT_DATA_DIR}")

    # --- 1. Chargement des fichiers bruts application_train et application_test ---
    try:
        with_target_df = pd.read_csv(os.path.join(INPUT_DATA_DIR, 'application_train.csv'), nrows=num_rows)
        without_target_df = pd.read_csv(os.path.join(INPUT_DATA_DIR, 'application_test.csv'), nrows=num_rows)
        print(f"application_train.csv chargé (lignes: {len(with_target_df)}), application_test.csv chargé (lignes: {len(without_target_df)})")
    except FileNotFoundError as e:
        print(f"ERREUR FATALE: Fichier CSV de base introuvable. Assurez-vous que 'application_train.csv' et 'application_test.csv' sont dans {INPUT_DATA_DIR}.")
        print(f"Détails: {e}")
        exit(1) # Arrête le script si les fichiers principaux ne sont pas là

    # Stocke les IDs et la cible pour la séparation après FE
    train_ids = with_target_df['SK_ID_CURR']
    train_target = with_target_df['TARGET']
    test_ids = without_target_df['SK_ID_CURR']

    # Concaténez les DataFrames pour appliquer le FE commun (important pour la cohérence des colonnes)
    # On retire la colonne 'TARGET' avant la concaténation, car elle n'existe pas dans le jeu de test.
    full_df = pd.concat([with_target_df.drop(columns=['TARGET']), without_target_df], ignore_index=True)
    full_df = full_df.reset_index(drop=True)
    del with_target_df, without_target_df # Libérer la mémoire
    gc.collect()

    # --- 2. Application du Feature Engineering sur les données 'application' ---
    print("Application du Feature Engineering sur les données 'application'...")
    # La ligne suivante peut générer un FutureWarning de Pandas, mais elle est fonctionnelle.
    full_df = full_df[full_df['CODE_GENDER'] != 'XNA']
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        full_df[bin_feature], _ = pd.factorize(full_df[bin_feature])
    full_df, _ = one_hot_encoder(full_df, nan_as_category)

    # La ligne suivante peut générer un FutureWarning de Pandas.
    full_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    full_df['DAYS_EMPLOYED_PERC'] = full_df['DAYS_EMPLOYED'] / full_df['DAYS_BIRTH']
    full_df['INCOME_CREDIT_PERC'] = np.where(full_df['AMT_CREDIT'] == 0, np.nan, full_df['AMT_INCOME_TOTAL'] / full_df['AMT_CREDIT'])
    full_df['INCOME_PER_PERSON'] = np.where(full_df['CNT_FAM_MEMBERS'] == 0, np.nan, full_df['AMT_INCOME_TOTAL'] / full_df['CNT_FAM_MEMBERS'])
    full_df['ANNUITY_INCOME_PERC'] = np.where(full_df['AMT_INCOME_TOTAL'] == 0, np.nan, full_df['AMT_ANNUITY'] / full_df['AMT_INCOME_TOTAL'])
    full_df['PAYMENT_RATE'] = np.where(full_df['AMT_CREDIT'] == 0, np.nan, full_df['AMT_ANNUITY'] / full_df['AMT_CREDIT'])
    print("Feature Engineering 'application' terminé.")

    # --- 3. TODO: Intégration des autres fichiers CSV et leur Feature Engineering ---
    # C'est la partie la plus critique et spécifique à VOTRE modèle.
    # Vous devez reproduire ici TOUTE la logique qui charge et fusionne
    # 'bureau.csv', 'bureau_balance.csv', 'previous_application.csv', etc.,
    # avec 'full_df' et applique leurs transformations (agrégations, one-hot, etc.).
    # Assurez-vous que ces fichiers sont présents dans votre dossier `input`.

    # EXEMPLE POUR 'bureau' et 'bureau_balance':
    # try:
    #     bureau_df = pd.read_csv(os.path.join(INPUT_DATA_DIR, 'bureau.csv'))
    #     bureau_balance_df = pd.read_csv(os.path.join(INPUT_DATA_DIR, 'bureau_balance.csv'))
    #     # Appelez ici votre fonction qui traite et fusionne les données de bureau. Ex:
    #     # full_df = process_bureau_data(full_df, bureau_df, bureau_balance_df)
    #     print("Fichiers 'bureau' traités et fusionnés (si la fonction est implémentée).")
    # except FileNotFoundError:
    #     print("AVERTISSEMENT: 'bureau.csv' ou 'bureau_balance.csv' introuvable. "
    #           "Si votre modèle les utilise, assurez-vous qu'ils sont dans le dossier d'entrée.")
    #     # Si ces fichiers sont obligatoires pour votre modèle, remplacez 'pass' par 'exit(1)'

    # Répétez le bloc try-except pour chaque fichier annexe (previous_application, POS_CASH_balance, etc.)
    # et leurs fonctions de traitement respectives.

    # --- 4. Séparation des DataFrames traités et ajout de la cible ---
    print("Séparation des données traitées en 'référence' et 'courantes'...")
    reference_data_processed = full_df[full_df['SK_ID_CURR'].isin(train_ids)].copy()
    current_data_processed = full_df[full_df['SK_ID_CURR'].isin(test_ids)].copy()

    # Ré-ajouter la colonne 'TARGET' au DataFrame de référence
    if 'TARGET' not in reference_data_processed.columns:
        original_train_for_target = pd.read_csv(os.path.join(INPUT_DATA_DIR, 'application_train.csv'), usecols=['SK_ID_CURR', 'TARGET'])
        reference_data_processed = reference_data_processed.merge(original_train_for_target, on='SK_ID_CURR', how='left')
    print("Données prêtes pour Evidently.")

    return reference_data_processed, current_data_processed


# --- Classe ColumnMapping Simplifiée pour Evidently 0.2.8 ---
# Cette classe est nécessaire car Evidently 0.2.8 ne permet pas d'importer ColumnMapping
# directement depuis evidently.model_profile ou evidently.column_mapping.
# Elle simule la structure attendue par Evidently pour le mapping des colonnes.
class SimplifiedColumnMapping:
    def __init__(self, target, prediction, numerical_features, categorical_features, datetime, id_feature):
        self.target = target
        self.prediction = prediction
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.datetime = datetime # Correction de l'AttributeError
        self.id_feature = id_feature


# --- Exécution Principale du Script ---
if __name__ == "__main__":
    print("Démarrage de la génération du rapport Evidently...")

    # Chargement et prétraitement des données
    reference_data, current_data = load_and_preprocess_data_for_evidently(num_rows=None, nan_as_category=False)
    print(f"Forme des données de référence (train traité): {reference_data.shape}")
    print(f"Forme des données courantes (test traité): {current_data.shape}")

    # --- Configuration du mapping des colonnes pour Evidently ---
    target_column_name = 'TARGET'
    id_column_name = 'SK_ID_CURR'

    # Détection automatique des features numériques et catégorielles après le Feature Engineering
    numerical_features_evidently = reference_data.select_dtypes(include=np.number).columns.tolist()
    categorical_features_evidently = reference_data.select_dtypes(include='object').columns.tolist()

    # Nettoyage : retirer la cible et l'ID des listes de features à analyser pour la dérive
    if target_column_name in numerical_features_evidently: numerical_features_evidently.remove(target_column_name)
    if target_column_name in categorical_features_evidently: categorical_features_evidently.remove(target_column_name)
    if id_column_name in numerical_features_evidently: numerical_features_evidently.remove(id_column_name)
    if id_column_name in categorical_features_evidently: categorical_features_evidently.remove(id_column_name)

    # Initialisation de notre ColumnMapping simplifiée
    column_mapping = SimplifiedColumnMapping(
        target=target_column_name,
        prediction=None, # Laissez None si vos données n'incluent pas les prédictions
        numerical_features=numerical_features_evidently,
        categorical_features=categorical_features_evidently,
        datetime=None, # Pas de colonne datetime spécifique pour l'analyse de dérive ici
        id_feature=id_column_name
    )

    # --- Création et exécution du rapport Evidently ---
    print("Création du rapport Evidently...")
    data_drift_report = Report(metrics=[
        DataDriftPreset(), # Ce préréglage inclut de nombreuses métriques pour la dérive des données
    ])

    data_drift_report.run(current_data=current_data, reference_data=reference_data, column_mapping=column_mapping)

    # Sauvegarde du rapport HTML
    print(f"Génération du rapport Evidently vers {OUTPUT_REPORT_PATH}...")
    data_drift_report.save_html(OUTPUT_REPORT_PATH)

    print("Rapport Evidently généré avec succès !")
    print(f"Vous pouvez trouver le rapport à l'emplacement suivant dans votre dossier partagé : C:\\Users\\jonjo\\Documents\\open classrooms\\Projet 7\\monitoring\\evidently_data_drift_report.html")