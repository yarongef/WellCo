"""
Configuration module for WellCo Churn Prediction
Contains all constants and configuration parameters
"""

import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("wellco_churn_analysis.log"),
        logging.StreamHandler(),
    ],
)


class ConfigConstants:
    """Model training and evaluation configuration"""

    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    RANDOM_STATE: int = 42
    RECENT_ACTIVITY_DAYS: int = 30
    DEFAULT_OUTREACH_SIZE: int = 200
    BASELINE_AUC: float = 0.501202
    CLASSIFICATION_THRESHOLD: float = 0.5  # Default threshold for classification
    THRESHOLD_OPTIMIZATION_METRIC: str = (
        "f1"  # Options: 'f1', 'balanced_accuracy', 'precision', 'recall'
    )


class BusinessConstants:
    """Business-specific constants and mappings"""

    CONTENT_CATEGORIES: Dict[str, List[str]] = {
        "nutrition": ["nutrition", "diet", "mediterranean", "fiber", "eating"],
        "exercise": ["exercise", "movement", "fitness", "cardio", "strength"],
        "sleep": ["sleep", "rest", "apnea"],
        "mental_health": ["stress", "mindfulness", "meditation", "wellness"],
    }

    # WellCo priority conditions (from client brief)
    PRIORITY_CONDITIONS = {
        "diabetes": "E11.9",  # Type 2 diabetes mellitus
        "hypertension": "I10",  # Essential hypertension
        "dietary_counseling": "Z71.3",  # Dietary counseling and surveillance
    }


class FileConstants:
    """File names and paths"""

    MODEL_FILE = "wellco_churn_model.pkl"
    RECOMMENDATIONS_FILE = "wellco_outreach_recommendations.csv"
    VISUALIZATIONS_FILE = "wellco_analysis_results.png"
    LOG_FILE = "wellco_churn_analysis.log"
