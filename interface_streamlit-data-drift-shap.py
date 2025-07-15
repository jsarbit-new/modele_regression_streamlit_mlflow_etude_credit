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

# --- 1. Page Configuration Streamlit ---
st.set_page_config(
    page_title="Prédiction de Défaut Client & Surveillance",
    page_icon="📊",
    layout="wide"
)

# --- 2. Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 3. MLflow Model Path Configuration ---
# IMPORTANT: Ce chemin doit pointer vers le dossier 'modele_mlflow'
# que vous avez copié de votre dossier mlruns/X/run_id/artifacts/model
# et placé à la racine de votre dépôt GitHub.
LOCAL_MODEL_PATH = "modele_mlflow"

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
# SHAP_IMPORTANT_FEATURES_NAMES = list(SHAP_IMPORTANT_FEATURES_INFO.keys()) # Cette ligne est redondante

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
# Cette ligne est déjà définie, pas besoin de la répéter
# SHAP_IMPORTANT_FEATURES_NAMES = list(SHAP_IMPORTANT_FEATURES_INFO.keys())

# --- Fonctions Stub pour la Génération de Données (utilisées pour la démo) ---
@st.cache_data(show_spinner="Chargement des données...")
def load_application_data_stub(num_rows):
    """
    Charge et retourne les données.
    Cette fonction ne s'exécutera qu'une seule fois.
    """
    try:
        # --- MODIFICATION ICI ---
        # Remplacez 'REMPLACEZ_PAR_VOTRE_FICHIER_CSV.csv' par le nom réel de votre fichier CSV
        # Par exemple: 'input/application_train.csv'
        df = pd.read_csv('input/application_train.csv') # Remplacez 'application_train.csv' par le nom de votre fichier 
        return df
    except FileNotFoundError:
        st.error("Fichier de données non trouvé. Assurez-vous que le fichier est bien à sa place dans le dossier 'input'.")
        return None

@st.cache_data(show_spinner="Traitement des données Bureau...")
def process_bureau_data_stub(df):
    for i in range(5):
        if f'bureau_feat_{i}' not in df.columns:
            df[f'bureau_feat_{i}'] = np.random.rand(len(df))
    return df

@st.cache_data(show_spinner="Traitement des demandes précédentes...")
def process_previous_applications_data_stub(df):
    for i in range(5):
        if f'prev_app_feat_{i}' not in df.columns:
            df[f'prev_app_feat_{i}'] = np.random.rand(len(df))
    return df

@st.cache_data(show_spinner="Traitement des données POS Cash...")
def process_pos_cash_data_stub(df):
    for i in range(5):
        if f'pos_feat_{i}' not in df.columns:
            df[f'pos_feat_{i}'] = np.random.rand(len(df))
    return df

@st.cache_data(show_spinner="Traitement des paiements d'acomptes...")
def process_installments_payments_data_stub(df):
    for i in range(5):
        if f'install_feat_{i}' not in df.columns:
            df[f'install_feat_{i}'] = np.random.rand(len(df))
    return df

@st.cache_data(show_spinner="Traitement des données de carte de crédit...")
def process_credit_card_balance_data_stub(df):
    for i in range(5):
        if f'cc_feat_{i}' not in df.columns:
            df[f'cc_feat_{i}'] = np.random.rand(len(df))
    return df

@st.cache_data(show_spinner="Exécution du pipeline d'ingénierie des caractéristiques...")
def run_feature_engineering_pipeline(num_rows):
    df = load_application_data_stub(num_rows)
    df = process_bureau_data_stub(df)
    df = process_previous_applications_data_stub(df)
    df = process_pos_cash_data_stub(df)
    df = process_installments_payments_data_stub(df)
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
    dummy_data = run_feature_engineering_pipeline(num_rows=1)
    all_training_features = list(dummy_data.columns)
    
    logger.info("Métadonnées du modèle chargées localement.")
    return features_info, optimal_threshold, all_training_features

@st.cache_resource(show_spinner="Chargement du pipeline du modèle...")
def load_mlflow_pipeline_local():
    """
    Charge le pipeline MLflow depuis le chemin local spécifié.
    """
    try:
        pipeline = mlflow.pyfunc.load_model(model_uri=LOCAL_MODEL_PATH)
        logger.info(f"Streamlit: Pipeline chargé depuis '{LOCAL_MODEL_PATH}'.")
        return pipeline
    except Exception as e:
        st.error(f"Échec lors du chargement du pipeline local: {e}")
        st.info(f"Assurez-vous que le dossier '{LOCAL_MODEL_PATH}' existe et contient un modèle MLflow valide.")
        return None

@st.cache_resource(show_spinner="Calcul de l'explainer SHAP...")
def load_shap_explainer(_pipeline, all_training_features):
    # Assurez-vous que le pipeline a bien un 'preprocessor'
    if 'preprocessor' in _pipeline.named_steps:
        preprocessor = _pipeline.named_steps['preprocessor']
    else:
        # Si le pipeline n'a pas de préprocesseur nommé 'preprocessor',
        # cela signifie que le modèle est peut-être directement le classifieur
        # ou que le préprocesseur est intégré différemment.
        # Pour cet exemple, nous allons simuler un préprocesseur si absent.
        logger.warning("Le pipeline ne contient pas de 'preprocessor' nommé. SHAP pourrait nécessiter un ajustement.")
        # Utiliser une identité si pas de préprocesseur explicite
        class IdentityPreprocessor:
            def transform(self, X):
                return X
            def get_feature_names_out(self):
                return X.columns
        preprocessor = IdentityPreprocessor()
    
    final_model = _pipeline.steps[-1][1] # Le dernier pas du pipeline est le modèle final
    
    # Génération de données de référence pour l'explainer SHAP
    # Assurez-vous que ces données sont représentatives de vos données d'entraînement
    ref_data_raw = run_feature_engineering_pipeline(num_rows=1000)
    
    # Assurez-vous que les colonnes de ref_data_raw correspondent à all_training_features
    # avant de les passer au préprocesseur
    ref_data_raw_filtered = ref_data_raw[all_training_features]
    
    ref_data_processed = preprocessor.transform(ref_data_raw_filtered)
    
    # Obtenir les noms de features après préprocessing
    # Si preprocessor.get_feature_names_out() n'existe pas ou ne fonctionne pas,
    # vous devrez peut-être les déduire manuellement ou les définir.
    try:
        processed_feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback si get_feature_names_out n'est pas disponible
        processed_feature_names = [f"col_{i}" for i in range(ref_data_processed.shape[1])]
        logger.warning("Impossible d'obtenir les noms de features du préprocesseur. Noms génériques utilisés.")

    ref_data_df = pd.DataFrame(ref_data_processed, columns=processed_feature_names)
    
    return shap.Explainer(final_model, ref_data_df)

@st.cache_data(show_spinner="Génération des données de référence pour le drift...")
def load_reference_data_for_drift():
    try:
        # Utilise la fonction stub pour générer des données de référence
        reference_df = run_feature_engineering_pipeline(num_rows=30000)
        logger.info(f"Données de référence chargées avec succès. Nombre d'échantillons: {len(reference_df)}")
        return reference_df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données de référence : {e}")
        st.stop()
        return None

# --- Fonctions d'Affichage des Rapports ---
def generate_and_display_evidently_report(reference_df, current_df):
    try:
        st.info("Génération du rapport en cours. Cela peut prendre quelques instants...")
        report_file_path = "evidently_data_drift_report_temp.html"
        data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
        data_drift_dashboard.calculate(reference_data=reference_df, current_data=current_df)
        data_drift_dashboard.save(report_file_path)
        with open(report_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=1000, scrolling=True)
        st.success("Rapport Evidently généré et affiché avec succès.")
        os.remove(report_file_path)
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
        # Assurez-vous que input_df contient toutes les colonnes attendues par le préprocesseur
        # et dans le bon ordre.
        # Si input_df n'a que les features importantes, il faut le compléter avec des zéros
        # ou des valeurs par défaut pour les autres features attendues par le préprocesseur.
        
        # Pour la démo, on s'assure que input_df a toutes les colonnes de all_training_features
        # avec des valeurs par défaut si elles ne sont pas dans user_inputs
        full_input_df = pd.DataFrame(columns=all_training_features)
        full_input_df.loc[0] = 0 # Initialise avec des zéros
        for col in input_df.columns:
            if col in full_input_df.columns:
                full_input_df[col] = input_df[col]

        input_for_shap = preprocessor.transform(full_input_df[all_training_features])
        shap_values_instance = shap_explainer(input_for_shap)
        
        processed_feature_names = preprocessor.get_feature_names_out()
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
# Utilisation des fonctions de chargement local
features_info, optimal_threshold, all_training_features = load_model_metadata_local()
pipeline = load_mlflow_pipeline_local()

# --- 7. Contenu Principal de la Page Streamlit ---
st.title("📊 Prédiction de Défaut Client & Surveillance du Modèle")

tab1, tab2 = st.tabs(["Prédiction de Prêt", "Analyse du Data Drift"])

with tab1:
    st.markdown("""
    Cette application vous permet de simuler une prédiction de risque de défaut pour un client.
    """)
    if features_info and pipeline: # Vérifie que le modèle et les métadonnées sont chargés
        st.sidebar.header("Informations sur le Modèle")
        st.sidebar.write(f"**Nom du Modèle :** `Pipeline de Régression Logistique`") # Nom générique
        st.sidebar.write(f"**Source du Modèle :** `Local (Dossier '{LOCAL_MODEL_PATH}')`")
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
                shap_explainer = load_shap_explainer(pipeline, all_training_features)
                preprocessor_for_shap = pipeline.named_steps['preprocessor']
                
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
    else:
        st.error("L'application n'a pas pu être initialisée. Vérifiez les logs pour plus de détails.")

with tab2:
    st.header("Analyse du Data Drift (Evidently AI)")
    st.markdown("""
    Cette section génère et affiche un rapport de **Data Drift** directement dans l'application.
    Le rapport compare les données d'entraînement (référence) aux données de production simulées.
    """)
    
    if st.button("Générer et afficher le rapport de Data Drift"):
        reference_data_for_drift = load_reference_data_for_drift()
        
        # Simulation de data drift
        df_production = reference_data_for_drift.copy()
        if 'AMT_CREDIT' in df_production.columns:
            df_production['AMT_CREDIT'] = df_production['AMT_CREDIT'] * np.random.normal(1.2, 0.1, len(df_production))
        if 'app_feature_17' in df_production.columns: # Correspond à l'âge client
            df_production['app_feature_17'] = df_production['app_feature_17'] + np.random.randint(-5, 5, len(df_production))
        
        generate_and_display_evidently_report(reference_data_for_drift, df_production)
    else:
        st.warning("Cliquez sur le bouton pour générer le rapport de dérive des données.")
