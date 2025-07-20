import pandas as pd
import pytest
from nouv_reg_logistique2 import MLFLOW_MODEL_NAME, run_feature_engineering_pipeline, find_best_threshold_and_fbeta
from interface_streamlit_data_drift_shap import load_mlflow_pipeline, load_mlflow_metadata

# Fixture pour charger le modèle MLflow
@pytest.fixture(scope="session")
def model_pipeline():
    """Charge le pipeline MLflow pour les tests."""
    try:
        pipeline = load_mlflow_pipeline(MLFLOW_MODEL_NAME)
        assert pipeline is not None
        return pipeline
    except Exception as e:
        pytest.fail(f"Échec du chargement du modèle MLflow: {e}")

# Test que le pipeline peut faire une prédiction
def test_pipeline_predict(model_pipeline):
    """Teste que le pipeline chargé peut faire une prédiction sur des données simulées."""
    input_df = run_feature_engineering_pipeline(debug_mode=True).head(1)
    
    # S'assurer que les features nécessaires sont présentes
    try:
        all_training_features = load_mlflow_metadata()[2]
    except Exception:
        pytest.fail("Impossible de charger les métadonnées du modèle. Assurez-vous d'avoir lancé le script d'entraînement au moins une fois.")
        
    assert all(col in input_df.columns for col in all_training_features)
    
    # Faire une prédiction et vérifier la sortie
    prediction_proba = model_pipeline.predict_proba(input_df.drop(columns=['TARGET', 'SK_ID_CURR']))
    
    assert prediction_proba.shape[0] == 1
    assert prediction_proba.shape[1] == 2
    assert isinstance(prediction_proba[0][0], float)

# Test que le calcul du seuil de F-beta fonctionne
def test_threshold_calculation():
    """Teste la fonction de recherche de seuil avec des données factices."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_probs = np.array([0.1, 0.3, 0.4, 0.6, 0.8, 0.9])
    
    best_threshold, best_fbeta = find_best_threshold_and_fbeta(y_true, y_probs, beta=2.0)
    
    assert best_threshold >= 0.59  # Le seuil devrait être près de 0.6
    assert best_fbeta > 0.9