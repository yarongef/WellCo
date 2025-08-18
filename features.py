"""
Feature engineering module for WellCo Churn Prediction
Handles data loading and feature creation
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Union, Optional
from pathlib import Path
import logging

from config import ConfigConstants, BusinessConstants


class FeatureEngineer:
    """
    Handles all feature engineering tasks including data loading and feature creation
    """

    def __init__(self, data_path: Union[str, Path] = ".") -> None:
        self.data_path: Union[str, Path] = data_path
        self.logger = logging.getLogger(__name__ + '.FeatureEngineer')

        # Data attributes
        self.churn_labels = None
        self.web_visits = None
        self.app_usage = None
        self.claims = None
        self.features = None

    def load_data(self) -> None:
        """Load all datasets"""
        self.logger.info("Loading datasets...")

        # Load main datasets
        self.churn_labels = pd.read_csv(f"{self.data_path}/churn_labels.csv")
        self.web_visits = pd.read_csv(f"{self.data_path}/web_visits.zip")
        self.app_usage = pd.read_csv(f"{self.data_path}/app_usage.csv")
        self.claims = pd.read_csv(f"{self.data_path}/claims.csv")

        self.logger.info(f"Loaded {len(self.churn_labels)} members")
        self.logger.info(f"Churn rate: {self.churn_labels['churn'].mean():.1%}")

    def create_temporal_features(self) -> pd.DataFrame:
        """Create time-based features from signup dates"""
        if self.churn_labels is None:
            raise ValueError("Churn labels data not loaded")

        self.churn_labels["signup_date"] = pd.to_datetime(
            self.churn_labels["signup_date"]
        )

        # Calculate days since signup (tenure)
        reference_date = pd.Timestamp.today().normalize()
        self.churn_labels["tenure_days"] = (
            reference_date - self.churn_labels["signup_date"]
        ).dt.days

        # Signup timing features
        self.churn_labels["signup_month"] = self.churn_labels["signup_date"].dt.month
        self.churn_labels["signup_year"] = self.churn_labels["signup_date"].dt.year

        return self.churn_labels[
            ["member_id", "tenure_days", "signup_month", "signup_year"]
        ].copy()

    def create_web_features(self) -> pd.DataFrame:
        """Create features from web visit data"""
        if self.web_visits is None or self.churn_labels is None:
            raise ValueError("Required data not loaded")

        if self.web_visits.empty:
            return pd.DataFrame({"member_id": self.churn_labels["member_id"]})

        # Convert timestamp
        self.web_visits["timestamp"] = pd.to_datetime(self.web_visits["timestamp"])

        # Basic engagement metrics
        web_features = (
            self.web_visits.groupby("member_id")
            .agg({"url": "count", "timestamp": ["min", "max"]})
            .reset_index()
        )

        web_features.columns = [
            "member_id",
            "total_visits",
            "first_visit",
            "last_visit",
        ]

        # Calculate engagement patterns
        web_features["days_active"] = (
            web_features["last_visit"] - web_features["first_visit"]
        ).dt.days + 1
        web_features["avg_visits_per_day"] = (
            web_features["total_visits"] / web_features["days_active"]
        )
        web_features["avg_visits_per_day"] = web_features["avg_visits_per_day"].fillna(
            0
        )

        # Content type analysis based on WellCo's focus areas
        for category, keywords in BusinessConstants.CONTENT_CATEGORIES.items():
            pattern = "|".join(keywords)
            category_visits: pd.DataFrame = (
                self.web_visits[
                    self.web_visits["title"].str.contains(pattern, case=False, na=False)
                    | self.web_visits["description"].str.contains(
                        pattern, case=False, na=False
                    )
                ]
                .groupby("member_id")["url"]
                .count()
                .reset_index()
            )
            category_visits.columns = ["member_id", f"{category}_visits"]
            web_features = web_features.merge(
                category_visits, on="member_id", how="left"
            )
            web_features[f"{category}_visits"] = web_features[
                f"{category}_visits"
            ].fillna(0)

        # Drop timestamp columns for final features
        web_features = web_features.drop(["first_visit", "last_visit"], axis=1)

        return web_features

    def create_app_features(self) -> pd.DataFrame:
        """Create features from app usage data"""
        if self.app_usage is None or self.churn_labels is None:
            raise ValueError("Required data not loaded")

        if self.app_usage.empty:
            return pd.DataFrame({"member_id": self.churn_labels["member_id"]})

        # Convert timestamp
        self.app_usage["timestamp"] = pd.to_datetime(self.app_usage["timestamp"])

        # Basic app engagement
        app_features = (
            self.app_usage.groupby("member_id")
            .agg({"event_type": "count", "timestamp": ["min", "max"]})
            .reset_index()
        )

        app_features.columns = [
            "member_id",
            "total_sessions",
            "first_session",
            "last_session",
        ]

        # Session patterns
        app_features["app_days_active"] = (
            app_features["last_session"] - app_features["first_session"]
        ).dt.days + 1
        app_features["avg_sessions_per_day"] = (
            app_features["total_sessions"] / app_features["app_days_active"]
        )
        app_features["avg_sessions_per_day"] = app_features[
            "avg_sessions_per_day"
        ].fillna(0)

        # Recent activity (last ConfigConstants.RECENT_ACTIVITY_DAYS days)
        recent_cutoff = self.app_usage["timestamp"].max() - timedelta(
            days=ConfigConstants.RECENT_ACTIVITY_DAYS
        )
        recent_sessions = (
            self.app_usage[self.app_usage["timestamp"] > recent_cutoff]
            .groupby("member_id")["event_type"]
            .count()
            .reset_index()
        )
        recent_sessions.columns = ["member_id", "recent_sessions"]
        app_features = app_features.merge(recent_sessions, on="member_id", how="left")
        app_features["recent_sessions"] = app_features["recent_sessions"].fillna(0)

        # Drop timestamp columns
        app_features = app_features.drop(["first_session", "last_session"], axis=1)

        return app_features

    def create_clinical_features(self) -> pd.DataFrame:
        """Create features from claims/clinical data"""
        if self.claims is None or self.churn_labels is None:
            raise ValueError("Required data not loaded")

        if self.claims.empty:
            return pd.DataFrame({"member_id": self.churn_labels["member_id"]})

        # Convert diagnosis date
        self.claims["diagnosis_date"] = pd.to_datetime(self.claims["diagnosis_date"])

        # Basic claims activity
        claims_features = (
            self.claims.groupby("member_id")
            .agg({"icd_code": "count", "diagnosis_date": ["min", "max"]})
            .reset_index()
        )

        claims_features.columns = [
            "member_id",
            "total_diagnoses",
            "first_diagnosis",
            "last_diagnosis",
        ]

        for condition, icd_code in BusinessConstants.PRIORITY_CONDITIONS.items():
            condition_flag: pd.DataFrame = (
                self.claims[self.claims["icd_code"] == icd_code]
                .groupby("member_id")["icd_code"]
                .count()
                .reset_index()
            )
            condition_flag.columns = ["member_id", f"has_{condition}"]
            condition_flag[f"has_{condition}"] = (
                condition_flag[f"has_{condition}"] > 0
            ).astype(int)
            claims_features = claims_features.merge(
                condition_flag, on="member_id", how="left"
            )
            claims_features[f"has_{condition}"] = claims_features[
                f"has_{condition}"
            ].fillna(0)

        # Comorbidity score (number of unique diagnosis codes)
        unique_diagnoses = (
            self.claims.groupby("member_id")["icd_code"].nunique().reset_index()
        )
        unique_diagnoses.columns = ["member_id", "comorbidity_score"]
        claims_features = claims_features.merge(
            unique_diagnoses, on="member_id", how="left"
        )

        # Drop timestamp columns
        claims_features = claims_features.drop(
            ["first_diagnosis", "last_diagnosis"], axis=1
        )

        return claims_features

    def engineer_features(self) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        self.logger.info("Engineering features...")

        # Create feature sets
        temporal_features = self.create_temporal_features()
        web_features = self.create_web_features()
        app_features = self.create_app_features()
        clinical_features = self.create_clinical_features()

        # Merge all features
        features = temporal_features
        feature_sets = [web_features, app_features, clinical_features]
        for feature_set in feature_sets:
            features = features.merge(feature_set, on="member_id", how="left")

        # Fill missing values (missing engagement data = zero engagement)
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        features[numeric_columns] = features[numeric_columns].fillna(0)

        # Add target and outreach flag
        if self.churn_labels is not None:
            features = features.merge(
                self.churn_labels[["member_id", "churn", "outreach"]],
                on="member_id",
                how="left",
            )

        self.features = features
        self.logger.info(
            f"Created {len(features.columns) - 3} features for {len(features)} members"
        )

        return features

    def get_features(self) -> pd.DataFrame:
        """Get the engineered features"""
        if self.features is None:
            raise ValueError("Features not generated. Run engineer_features() first.")
        return self.features