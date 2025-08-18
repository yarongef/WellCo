"""
Evaluation module for WellCo Churn Prediction
Handles model evaluation, visualization, and reporting with optimal threshold analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import logging

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)

from config import ConfigConstants, FileConstants


class Evaluator:
    """
    Handles model evaluation, visualization, and reporting with optimal threshold analysis
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__ + '.Evaluator')

    def create_visualizations(
        self,
        results: Dict[str, Dict[str, Any]],
        y_test: pd.Series,
        model_name: str,
        features: pd.DataFrame,
        confusion_mat_default: Optional[np.ndarray] = None,
        confusion_mat_optimal: Optional[np.ndarray] = None,
    ) -> None:
        """Create comprehensive visualizations including threshold comparison"""
        plt.style.use("default")

        # Create subplot layout with additional plots for threshold analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # Model comparison
        models = list(results.keys())
        auc_scores = [results[model]["auc_test"] for model in models]

        axes[0].bar(
            models,
            auc_scores,
            color=["skyblue", "lightgreen", "lightcoral", "lightyellow"],
        )
        axes[0].axhline(
            y=ConfigConstants.BASELINE_AUC,
            color="red",
            linestyle="--",
            label="Baseline",
        )
        axes[0].set_title("Model Performance Comparison (AUC)")
        axes[0].set_ylabel("AUC Score")
        axes[0].legend()
        axes[0].tick_params(axis="x", rotation=45)

        # ROC Curve
        best_predictions = results[model_name]["predictions"]
        fpr, tpr, _ = roc_curve(y_test, best_predictions)

        axes[1].plot(
            fpr,
            tpr,
            label=f'{model_name} (AUC = {results[model_name]["auc_test"]:.3f})',
        )
        axes[1].plot([0, 1], [0, 1], "k--", label="Random")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend()

        # Precision-Recall Curve
        precision, recall, thresholds = precision_recall_curve(y_test, best_predictions)
        axes[2].plot(recall, precision, label=f"{model_name}")
        axes[2].set_xlabel("Recall")
        axes[2].set_ylabel("Precision")
        axes[2].set_title("Precision-Recall Curve")

        # Mark optimal threshold point
        optimal_threshold = results[model_name]["optimal_threshold"]
        optimal_recall = results[model_name]["recall_optimal"]
        optimal_precision = results[model_name]["precision_optimal"]
        axes[2].plot(
            optimal_recall,
            optimal_precision,
            "ro",
            markersize=8,
            label=f"Optimal Threshold ({optimal_threshold:.3f})",
        )
        axes[2].legend()

        # Threshold Analysis
        thresholds_range = np.arange(0.1, 0.9, 0.01)
        f1_scores = []
        precision_scores = []
        recall_scores = []

        for threshold in thresholds_range:
            y_pred_thresh = (best_predictions >= threshold).astype(int)
            f1_scores.append(f1_score(y_test, y_pred_thresh))
            precision_scores.append(
                precision_score(y_test, y_pred_thresh, zero_division=0)
            )
            recall_scores.append(recall_score(y_test, y_pred_thresh))

        axes[3].plot(thresholds_range, f1_scores, label="F1 Score", color="blue")
        axes[3].plot(
            thresholds_range, precision_scores, label="Precision", color="green"
        )
        axes[3].plot(thresholds_range, recall_scores, label="Recall", color="orange")
        axes[3].axvline(
            x=optimal_threshold,
            color="red",
            linestyle="--",
            label=f"Optimal Threshold ({optimal_threshold:.3f})",
        )
        axes[3].set_xlabel("Threshold")
        axes[3].set_ylabel("Score")
        axes[3].set_title("Metrics vs Threshold")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        # Confusion Matrix Comparison
        if confusion_mat_default is not None and confusion_mat_optimal is not None:
            # Default threshold confusion matrix
            im1 = axes[4].imshow(
                confusion_mat_default, interpolation="nearest", cmap=plt.cm.Blues
            )
            axes[4].set_title("Confusion Matrix\n(Default Threshold: 0.5)")

            # Add text annotations
            thresh = confusion_mat_default.max() / 2.0
            for i in range(confusion_mat_default.shape[0]):
                for j in range(confusion_mat_default.shape[1]):
                    axes[4].text(
                        j,
                        i,
                        format(confusion_mat_default[i, j], "d"),
                        ha="center",
                        va="center",
                        color=(
                            "white" if confusion_mat_default[i, j] > thresh else "black"
                        ),
                    )

            axes[4].set_xlabel("Predicted Label")
            axes[4].set_ylabel("True Label")
            axes[4].set_xticks([0, 1])
            axes[4].set_yticks([0, 1])
            axes[4].set_xticklabels(["No Churn", "Churn"])
            axes[4].set_yticklabels(["No Churn", "Churn"])

            # Optimal threshold confusion matrix
            im2 = axes[5].imshow(
                confusion_mat_optimal, interpolation="nearest", cmap=plt.cm.Blues
            )
            axes[5].set_title(
                f"Confusion Matrix\n(Optimal Threshold: {optimal_threshold:.3f})"
            )

            # Add text annotations
            thresh = confusion_mat_optimal.max() / 2.0
            for i in range(confusion_mat_optimal.shape[0]):
                for j in range(confusion_mat_optimal.shape[1]):
                    axes[5].text(
                        j,
                        i,
                        format(confusion_mat_optimal[i, j], "d"),
                        ha="center",
                        va="center",
                        color=(
                            "white" if confusion_mat_optimal[i, j] > thresh else "black"
                        ),
                    )

            axes[5].set_xlabel("Predicted Label")
            axes[5].set_ylabel("True Label")
            axes[5].set_xticks([0, 1])
            axes[5].set_yticks([0, 1])
            axes[5].set_xticklabels(["No Churn", "Churn"])
            axes[5].set_yticklabels(["No Churn", "Churn"])
        else:
            # Churn distribution if no confusion matrices provided
            churn_counts = features["churn"].value_counts()
            axes[4].pie(
                churn_counts.values, labels=["No Churn", "Churn"], autopct="%1.1f%%"
            )
            axes[4].set_title("Churn Distribution")

            # Hide the last subplot if not needed
            axes[5].axis("off")

        plt.tight_layout()
        plt.savefig(FileConstants.VISUALIZATIONS_FILE, dpi=300, bbox_inches="tight")
        plt.show()

        self.logger.info(f"Comprehensive visualizations saved to '{FileConstants.VISUALIZATIONS_FILE}'")

    def print_analysis_summary(
        self,
        results: Dict[str, Dict[str, Any]],
        model_name: str,
        optimal_n: int,
        predictions_df: pd.DataFrame,
        classification_rep_default: str,
        classification_rep_optimal: str,
    ) -> None:
        """Print comprehensive analysis summary with threshold comparison"""
        best_results = results[model_name]
        optimal_threshold = best_results["optimal_threshold"]

        self.logger.info("\n" + "=" * 80)
        self.logger.info("COMPREHENSIVE ANALYSIS SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Best Model: {model_name}")
        self.logger.info(f"Test AUC: {best_results['auc_test']:.4f}")
        self.logger.info(f"Baseline AUC: {ConfigConstants.BASELINE_AUC:.4f}")
        self.logger.info(
            f"Improvement: +{best_results['auc_test'] - ConfigConstants.BASELINE_AUC:.4f}"
        )
        self.logger.info(
            f"Optimal Threshold: {optimal_threshold:.3f} (optimized for {ConfigConstants.THRESHOLD_OPTIMIZATION_METRIC})"
        )
        self.logger.info(f"Recommended outreach: {optimal_n} members")
        self.logger.info(
            f"Expected churn rate in top {optimal_n}: {predictions_df.head(optimal_n)['actual_churn'].mean():.1%}"
        )

        self.logger.info("\n" + "=" * 80)
        self.logger.info("THRESHOLD COMPARISON")
        self.logger.info("=" * 80)

        # Create comparison table
        header = f"{'Metric':<20} {'Default (0.5)':<15} {'Optimal ({:.3f})':<15} {'Improvement':<12}".format(
            optimal_threshold
        )
        self.logger.info(header)
        self.logger.info("-" * 65)

        metrics = [
            (
                "Balanced Accuracy",
                "balanced_accuracy_default",
                "balanced_accuracy_optimal",
            ),
            ("F1 Score", "f1_default", "f1_optimal"),
            ("Precision", "precision_default", "precision_optimal"),
            ("Recall", "recall_default", "recall_optimal"),
        ]

        for metric_name, default_key, optimal_key in metrics:
            default_val = best_results[default_key]
            optimal_val = best_results[optimal_key]
            improvement = optimal_val - default_val
            improvement_str = (
                f"+{improvement:.4f}" if improvement >= 0 else f"{improvement:.4f}"
            )

            metric_line = f"{metric_name:<20} {default_val:<15.4f} {optimal_val:<15.4f} {improvement_str:<12}"
            self.logger.info(metric_line)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("CLASSIFICATION REPORT - DEFAULT THRESHOLD (0.5)")
        self.logger.info("=" * 80)
        self.logger.info(f"\n{classification_rep_default}")

        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"CLASSIFICATION REPORT - OPTIMAL THRESHOLD ({optimal_threshold:.3f})")
        self.logger.info("=" * 80)
        self.logger.info(f"\n{classification_rep_optimal}")

        self.logger.info("\n" + "=" * 80)
        self.logger.info("FILES GENERATED")
        self.logger.info("=" * 80)
        self.logger.info(
            f"• {FileConstants.RECOMMENDATIONS_FILE} - Top recommendations with optimal threshold predictions"
        )
        self.logger.info(
            f"• {FileConstants.MODEL_FILE} - Trained model with optimal threshold information"
        )
        self.logger.info(
            f"• {FileConstants.VISUALIZATIONS_FILE} - Comprehensive visualizations including threshold analysis"
        )
        self.logger.info(
            f"• {FileConstants.LOG_FILE} - Complete analysis log file"
        )
