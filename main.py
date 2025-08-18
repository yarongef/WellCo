"""
Main execution module for WellCo Churn Prediction Analysis
Orchestrates the complete analysis pipeline with optimal threshold selection
"""
from typing import Union
from pathlib import Path
import logging

from features import FeatureEngineer
from models import ModelTrainer
from evaluation import Evaluator


class WellCoChurnPredictor:
    """
    Main orchestrator class that coordinates the feature engineering, model training, and evaluation
    """

    def __init__(self, data_path: Union[str, Path] = ".") -> None:
        self.feature_engineer = FeatureEngineer(data_path)
        self.model_trainer = ModelTrainer()
        self.evaluator = Evaluator()
        self.logger = logging.getLogger(__name__ + '.WellCoChurnPredictor')

    def run_complete_analysis(self) -> None:
        """Execute the complete analysis pipeline with optimal threshold selection"""
        self.logger.info("=" * 80)
        self.logger.info("WellCo Churn Prediction Analysis with Optimal Threshold Selection")
        self.logger.info("=" * 80)

        # Load and process data
        self.feature_engineer.load_data()
        features = self.feature_engineer.engineer_features()

        # Train models with optimal threshold selection
        results, y_test = self.model_trainer.train_models(features)

        # Generate predictions using optimal threshold
        predictions_df, optimal_n = self.model_trainer.generate_predictions(features)

        # Save results with optimal threshold information
        self.model_trainer.save_model(predictions_df, optimal_n)

        # Get model info
        model_info = self.model_trainer.get_model_info()
        model_name = model_info["model_name"]

        # Generate classification reports for both thresholds
        classification_report_default = self.model_trainer.get_classification_report(
            results, use_optimal_threshold=False
        )
        classification_report_optimal = self.model_trainer.get_classification_report(
            results, use_optimal_threshold=True
        )

        # Generate confusion matrices for both thresholds
        confusion_mat_default = self.model_trainer.get_confusion_matrix(
            results, use_optimal_threshold=False
        )
        confusion_mat_optimal = self.model_trainer.get_confusion_matrix(
            results, use_optimal_threshold=True
        )

        # Create comprehensive visualizations
        self.evaluator.create_visualizations(
            results,
            y_test,
            model_name,
            features,
            confusion_mat_default,
            confusion_mat_optimal,
        )

        # Print comprehensive summary
        self.evaluator.print_analysis_summary(
            results,
            model_name,
            optimal_n,
            predictions_df,
            classification_report_default,
            classification_report_optimal,
        )


def main() -> None:
    """Main execution function"""
    # Initialize predictor
    predictor = WellCoChurnPredictor()

    # Run complete analysis with optimal threshold selection
    predictor.run_complete_analysis()


if __name__ == "__main__":
    main()
