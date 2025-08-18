"""
Model training and prediction module for WellCo Churn Prediction
Handles model training, selection, and prediction generation with optimal threshold selection
"""
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from config import ConfigConstants, FileConstants


class ModelTrainer:
    """
    Handles model training, selection, and prediction generation with optimal threshold selection
    """

    def __init__(self) -> None:
        self.model: Optional[BaseEstimator] = None
        self.model_name: Optional[str] = None
        self.feature_names: Optional[List[str]] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        self.optimal_threshold: Optional[float] = None
        self.logger = logging.getLogger(__name__ + '.ModelTrainer')

    def create_temporal_split(
        self, features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create temporal split based on signup_month and signup_year"""
        # Sort by signup year and month to create temporal ordering
        features_sorted = features.sort_values(
            ["signup_year", "signup_month"]
        ).reset_index(drop=True)

        # Calculate split point to get approximately 80/20 split
        split_idx = int(len(features_sorted) * (1 - ConfigConstants.TEST_SIZE))

        # Separate features and target
        feature_cols: List[str] = [
            col
            for col in features.columns
            if col not in ["member_id", "churn", "outreach", "signup_month", "signup_year"]
        ]

        X = features_sorted[feature_cols]
        y = features_sorted["churn"]

        # Store feature names
        self.feature_names = feature_cols

        # Create temporal split
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        # Store test data for later use
        self.X_test = X_test
        self.y_test = y_test

        self.logger.info(f"Temporal split created:")
        self.logger.info(f"Training set: {len(X_train)} samples")
        self.logger.info(f"Test set: {len(X_test)} samples")
        self.logger.info(f"Training churn rate: {y_train.mean():.1%}")
        self.logger.info(f"Test churn rate: {y_test.mean():.1%}")

        return X_train, X_test, y_train, y_test

    def _prepare_model_data(
        self, features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for modeling - returns unscaled data for pipelines"""
        # Use temporal split instead of random split
        return self.create_temporal_split(features)

    def _find_optimal_threshold(
        self, y_true: pd.Series, y_proba: np.ndarray, metric: str = "f1"
    ) -> Tuple[float, Dict[str, float]]:
        """Find optimal threshold based on specified metric"""
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = {}

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            # Calculate various metrics
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)

            scores[threshold] = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "balanced_accuracy": balanced_acc,
            }

        # Find optimal threshold based on chosen metric
        optimal_threshold = max(scores.keys(), key=lambda k: scores[k][metric])
        optimal_scores = scores[optimal_threshold]

        return optimal_threshold, optimal_scores

    def train_models(
        self, features: pd.DataFrame
    ) -> Tuple[Dict[str, Dict[str, Any]], pd.Series]:
        """Train and evaluate multiple models with proper pipelines and optimal threshold selection"""
        self.logger.info("Training models...")

        X_train, X_test, y_train, y_test = self._prepare_model_data(features)

        # Define models - using pipelines for models that need scaling
        models = {
            "Logistic Regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            random_state=ConfigConstants.RANDOM_STATE, max_iter=1000
                        ),
                    ),
                ]
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=ConfigConstants.RANDOM_STATE,
                class_weight="balanced",
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=ConfigConstants.RANDOM_STATE,
            ),
            "MLP": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        MLPClassifier(
                            hidden_layer_sizes=(100, 50),
                            activation="relu",
                            solver="adam",
                            alpha=0.001,
                            learning_rate="adaptive",
                            max_iter=1000,
                            random_state=ConfigConstants.RANDOM_STATE,
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=10,
                        ),
                    ),
                ]
            ),
        }

        results = {}

        for name, model in models.items():
            self.logger.info(f"Training {name}...")

            # Train model (pipeline handles scaling automatically)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Find optimal threshold
            optimal_threshold, optimal_scores = self._find_optimal_threshold(
                y_test, y_pred_proba, ConfigConstants.THRESHOLD_OPTIMIZATION_METRIC
            )

            # Predictions with default threshold
            y_pred_default = model.predict(X_test)

            # Predictions with optimal threshold
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

            # Calculate metrics for both thresholds
            auc_score = roc_auc_score(y_test, y_pred_proba)

            # Default threshold metrics
            balanced_acc_default = balanced_accuracy_score(y_test, y_pred_default)
            f1_default = f1_score(y_test, y_pred_default)
            precision_default = precision_score(y_test, y_pred_default, zero_division=0)
            recall_default = recall_score(y_test, y_pred_default)

            # Optimal threshold metrics
            balanced_acc_optimal = balanced_accuracy_score(y_test, y_pred_optimal)
            f1_optimal = f1_score(y_test, y_pred_optimal)
            precision_optimal = precision_score(y_test, y_pred_optimal, zero_division=0)
            recall_optimal = recall_score(y_test, y_pred_optimal)

            # Cross-validation (pipeline ensures proper scaling in each fold)
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=StratifiedKFold(
                    n_splits=ConfigConstants.CV_FOLDS,
                    shuffle=True,
                    random_state=ConfigConstants.RANDOM_STATE,
                ),
                scoring="roc_auc",
            )

            results[name] = {
                "model": model,
                "auc_test": auc_score,
                "optimal_threshold": optimal_threshold,
                # Default threshold results
                "balanced_accuracy_default": balanced_acc_default,
                "f1_default": f1_default,
                "precision_default": precision_default,
                "recall_default": recall_default,
                "predictions_binary_default": y_pred_default,
                # Optimal threshold results
                "balanced_accuracy_optimal": balanced_acc_optimal,
                "f1_optimal": f1_optimal,
                "precision_optimal": precision_optimal,
                "recall_optimal": recall_optimal,
                "predictions_binary_optimal": y_pred_optimal,
                # Common results
                "auc_cv_mean": cv_scores.mean(),
                "auc_cv_std": cv_scores.std(),
                "predictions": y_pred_proba,
            }

            self.logger.info(f"Test AUC: {auc_score:.4f}")
            self.logger.info(f"Optimal Threshold: {optimal_threshold:.3f}")
            self.logger.info(
                f"Default Threshold (0.5) - Balanced Acc: {balanced_acc_default:.4f}, F1: {f1_default:.4f}"
            )
            self.logger.info(
                f"Optimal Threshold ({optimal_threshold:.3f}) - Balanced Acc: {balanced_acc_optimal:.4f}, F1: {f1_optimal:.4f}"
            )
            self.logger.info(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Select best model based on AUC
        best_model_name = max(results.keys(), key=lambda k: results[k]["auc_test"])
        self.model = results[best_model_name]["model"]
        self.model_name = best_model_name
        self.optimal_threshold = results[best_model_name]["optimal_threshold"]

        self.logger.info(
            f"Best model: {best_model_name} (AUC: {results[best_model_name]['auc_test']:.4f})"
        )
        self.logger.info(f"Optimal threshold: {self.optimal_threshold:.3f}")
        self.logger.info(
            f"Improvement over baseline: {results[best_model_name]['auc_test'] - ConfigConstants.BASELINE_AUC:.4f}"
        )

        return results, y_test

    def generate_predictions(
        self, features: pd.DataFrame, n_members: Optional[int] = None
    ) -> Tuple[pd.DataFrame, int]:
        """Generate ranked list of members for outreach using optimal threshold"""
        if self.model is None:
            raise ValueError("Model not trained")

        # Get feature data
        feature_cols = [
            col
            for col in features.columns
            if col not in ["member_id", "churn", "outreach", "signup_month", "signup_year"]
        ]
        X = features[feature_cols]

        # Generate predictions (pipeline handles scaling automatically)
        churn_probabilities = self.model.predict_proba(X)[:, 1]

        # Use optimal threshold for binary classification
        churn_predictions_optimal = (
            churn_probabilities >= self.optimal_threshold
        ).astype(int)

        # Create predictions dataframe
        predictions_df: pd.DataFrame = pd.DataFrame(
            {
                "member_id": features["member_id"],
                "churn_probability": churn_probabilities,
                "churn_prediction_optimal": churn_predictions_optimal,
                "actual_churn": features["churn"],
                "received_outreach": features["outreach"],
            }
        )

        # Rank by churn probability
        predictions_df = predictions_df.sort_values(
            "churn_probability", ascending=False
        )
        predictions_df["rank"] = range(1, len(predictions_df) + 1)

        # Determine optimal n if not provided
        if n_members is None:
            n_members = self._determine_optimal_n(predictions_df, features)

        # Get top n recommendations
        top_n_predictions = predictions_df.head(n_members)

        return top_n_predictions, n_members

    def _determine_optimal_n(
        self,
        predictions_df: pd.DataFrame,
        features: pd.DataFrame,
        cost_per_outreach: float = 10,
        revenue_per_retained: float = 1000,
        retention_rate: float = 0.3,
    ) -> int:
        """Determine optimal number for outreach using cost-benefit analysis on training data only"""

        # Get training data predictions only (exclude test data)
        # Split point calculation matches create_temporal_split
        features_sorted = features.sort_values(
            ["signup_year", "signup_month"]
        ).reset_index(drop=True)
        split_idx = int(len(features_sorted) * (1 - ConfigConstants.TEST_SIZE))

        # Get member_ids from training set
        train_member_ids = set(features_sorted.iloc[:split_idx]["member_id"])

        # Filter predictions to only include training data
        train_predictions_df = predictions_df[
            predictions_df["member_id"].isin(train_member_ids)
        ].copy()
        train_predictions_df = train_predictions_df.sort_values(
            "churn_probability", ascending=False
        ).reset_index(drop=True)

        self.logger.info(
            f"Using {len(train_predictions_df)} training samples for optimal n determination"
        )

        # Test different n values
        n_values = range(50, min(1500, len(train_predictions_df)), 50)
        best_n = ConfigConstants.DEFAULT_OUTREACH_SIZE
        best_profit = -float("inf")

        for n in n_values:
            top_n = train_predictions_df.head(n)
            actual_churners = top_n["actual_churn"].sum()  # True positives

            # Calculate costs and benefits
            total_cost = n * cost_per_outreach
            retained_customers = actual_churners * retention_rate
            total_revenue = retained_customers * revenue_per_retained
            profit = total_revenue - total_cost

            # Track best option
            if profit > best_profit:
                best_profit = profit
                best_n = n

        # Get final metrics from training data
        top_n_train = train_predictions_df.head(best_n)
        precision_train = top_n_train["actual_churn"].mean()

        self.logger.info(f"Optimal n (determined from training data): {best_n} members")
        self.logger.info(f"Training precision at optimal n: {precision_train:.1%}")
        self.logger.info(f"Expected profit (from training analysis): ${best_profit:,.0f}")

        return best_n

    def get_classification_report(
        self, results: Dict[str, Dict[str, Any]], use_optimal_threshold: bool = True
    ) -> str:
        """Generate classification report for the best model"""
        if self.model is None or self.y_test is None:
            raise ValueError("Model not trained or test data not available")

        best_model_name = self.model_name

        if use_optimal_threshold:
            y_pred_binary = results[best_model_name]["predictions_binary_optimal"]
            threshold_used = results[best_model_name]["optimal_threshold"]
        else:
            y_pred_binary = results[best_model_name]["predictions_binary_default"]
            threshold_used = 0.5

        # Generate classification report
        class_names = ["No Churn", "Churn"]
        report = classification_report(
            self.y_test, y_pred_binary, target_names=class_names, digits=4
        )

        report_with_header = f"Classification Report (Threshold: {threshold_used:.3f})\n{'='*60}\n{report}"

        return report_with_header

    def get_confusion_matrix(
        self, results: Dict[str, Dict[str, Any]], use_optimal_threshold: bool = True
    ) -> np.ndarray:
        """Generate confusion matrix for the best model"""
        if self.model is None or self.y_test is None:
            raise ValueError("Model not trained or test data not available")

        best_model_name = self.model_name

        if use_optimal_threshold:
            y_pred_binary = results[best_model_name]["predictions_binary_optimal"]
        else:
            y_pred_binary = results[best_model_name]["predictions_binary_default"]

        return confusion_matrix(self.y_test, y_pred_binary)

    def save_model(self, predictions_df: pd.DataFrame, n_members: int) -> None:
        """Save model and predictions with optimal threshold information"""
        if self.model is None:
            raise ValueError("No model to save")

        # Save model (pipeline includes scaler)
        model_info = {
            "model": self.model,
            "optimal_threshold": self.optimal_threshold,
            "feature_names": self.feature_names,
            "model_name": self.model_name,
        }
        joblib.dump(model_info, FileConstants.MODEL_FILE)
        self.logger.info(f"Model saved with optimal threshold {self.optimal_threshold:.3f}")

        # Save top n predictions
        output_df = predictions_df.head(n_members)[
            ["member_id", "churn_probability", "churn_prediction_optimal", "rank"]
        ].copy()
        output_df.columns = [
            "member_id",
            "prioritization_score",
            "churn_prediction",
            "rank",
        ]
        output_df.to_csv(FileConstants.RECOMMENDATIONS_FILE, index=False)

        self.logger.info(
            f"Saved top {n_members} recommendations to '{FileConstants.RECOMMENDATIONS_FILE}'"
        )
        self.logger.info(f"Saved model with optimal threshold to '{FileConstants.MODEL_FILE}'")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        return {
            "model": self.model,
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "optimal_threshold": self.optimal_threshold,
        }