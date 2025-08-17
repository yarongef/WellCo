import pandas as pd
from datetime import timedelta

class WellCoChurnPredictor:
    """
    Main class for WellCo churn prediction analysis
    """

    def __init__(self, data_path='.'):
        self.data_path = data_path

    def load_data(self):
        """Load all datasets"""
        print("Loading datasets...")

        # Load main datasets
        self.churn_labels = pd.read_csv(f'{self.data_path}/churn_labels.csv')
        self.web_visits = pd.read_csv(f'{self.data_path}/web_visits.zip')
        self.app_usage = pd.read_csv(f'{self.data_path}/app_usage.csv')
        self.claims = pd.read_csv(f'{self.data_path}/claims.csv')

        print(f"Loaded {len(self.churn_labels)} members")
        print(f"Churn rate: {self.churn_labels['churn'].mean():.1%}")

    def create_temporal_features(self):
        """Create time-based features from signup dates"""
        self.churn_labels['signup_date'] = pd.to_datetime(self.churn_labels['signup_date'])

        # Calculate days since signup (tenure)
        reference_date = self.churn_labels['signup_date'].max()
        self.churn_labels['tenure_days'] = (reference_date - self.churn_labels['signup_date']).dt.days

        # Signup timing features
        self.churn_labels['signup_month'] = self.churn_labels['signup_date'].dt.month
        self.churn_labels['signup_quarter'] = self.churn_labels['signup_date'].dt.quarter
        self.churn_labels['signup_year'] = self.churn_labels['signup_date'].dt.year

        return self.churn_labels[['member_id', 'tenure_days', 'signup_month',
                                  'signup_quarter', 'signup_year']].copy()

    def create_web_features(self):
        """Create features from web visit data"""
        if self.web_visits.empty: #todo: remove
            return pd.DataFrame({'member_id': self.churn_labels['member_id']})

        # Convert timestamp
        self.web_visits['timestamp'] = pd.to_datetime(self.web_visits['timestamp'])

        # Basic engagement metrics
        web_features = self.web_visits.groupby('member_id').agg({
            'url': 'count',
            'timestamp': ['min', 'max']
        }).reset_index()

        web_features.columns = ['member_id', 'total_visits', 'first_visit', 'last_visit']

        # Calculate engagement patterns
        web_features['days_active'] = (web_features['last_visit'] - web_features['first_visit']).dt.days + 1
        web_features['avg_visits_per_day'] = web_features['total_visits'] / web_features['days_active']
        web_features['avg_visits_per_day'] = web_features['avg_visits_per_day'].fillna(0)

        # Content type analysis based on WellCo's focus areas
        content_keywords = {
            'nutrition': ['nutrition', 'diet', 'mediterranean', 'fiber', 'eating'],
            'exercise': ['exercise', 'movement', 'fitness', 'cardio', 'strength'],
            'sleep': ['sleep', 'rest', 'apnea'],
            'mental_health': ['stress', 'mindfulness', 'meditation', 'wellness']
        }

        for category, keywords in content_keywords.items():
            pattern = '|'.join(keywords)
            category_visits = self.web_visits[
                self.web_visits['title'].str.contains(pattern, case=False, na=False) |
                self.web_visits['description'].str.contains(pattern, case=False, na=False)
                ].groupby('member_id')['url'].count().reset_index()
            category_visits.columns = ['member_id', f'{category}_visits']
            web_features = web_features.merge(category_visits, on='member_id', how='left')
            web_features[f'{category}_visits'] = web_features[f'{category}_visits'].fillna(0)

        # Drop timestamp columns for final features
        web_features = web_features.drop(['first_visit', 'last_visit'], axis=1)

        return web_features

    def create_app_features(self):
        """Create features from app usage data"""
        if self.app_usage.empty:
            return pd.DataFrame({'member_id': self.churn_labels['member_id']})

        # Convert timestamp
        self.app_usage['timestamp'] = pd.to_datetime(self.app_usage['timestamp'])

        # Basic app engagement
        app_features = self.app_usage.groupby('member_id').agg({
            'event_type': 'count',
            'timestamp': ['min', 'max']
        }).reset_index()

        app_features.columns = ['member_id', 'total_sessions', 'first_session', 'last_session']

        # Session patterns
        app_features['app_days_active'] = (app_features['last_session'] - app_features['first_session']).dt.days + 1
        app_features['avg_sessions_per_day'] = app_features['total_sessions'] / app_features['app_days_active']
        app_features['avg_sessions_per_day'] = app_features['avg_sessions_per_day'].fillna(0)

        # Recent activity (last 30 days)
        recent_cutoff = self.app_usage['timestamp'].max() - timedelta(days=30)
        recent_sessions = self.app_usage[self.app_usage['timestamp'] > recent_cutoff].groupby('member_id')[
            'event_type'].count().reset_index()
        recent_sessions.columns = ['member_id', 'recent_sessions']
        app_features = app_features.merge(recent_sessions, on='member_id', how='left')
        app_features['recent_sessions'] = app_features['recent_sessions'].fillna(0)

        # Drop timestamp columns
        app_features = app_features.drop(['first_session', 'last_session'], axis=1)

        return app_features

    def create_clinical_features(self):
        """Create features from claims/clinical data"""
        if self.claims.empty:
            return pd.DataFrame({'member_id': self.churn_labels['member_id']})

        # Convert diagnosis date
        self.claims['diagnosis_date'] = pd.to_datetime(self.claims['diagnosis_date'])

        # Basic claims activity
        claims_features = self.claims.groupby('member_id').agg({
            'icd_code': 'count',
            'diagnosis_date': ['min', 'max']
        }).reset_index()

        claims_features.columns = ['member_id', 'total_diagnoses', 'first_diagnosis', 'last_diagnosis']

        # WellCo priority conditions (from client brief)
        priority_conditions = {
            'diabetes': 'E11.9',  # Type 2 diabetes mellitus
            'hypertension': 'I10',  # Essential hypertension
            'dietary_counseling': 'Z71.3'  # Dietary counseling and surveillance
        }

        for condition, icd_code in priority_conditions.items():
            condition_flag = self.claims[self.claims['icd_code'] == icd_code].groupby('member_id')[
                'icd_code'].count().reset_index()
            condition_flag.columns = ['member_id', f'has_{condition}']
            condition_flag[f'has_{condition}'] = (condition_flag[f'has_{condition}'] > 0).astype(int)
            claims_features = claims_features.merge(condition_flag, on='member_id', how='left')
            claims_features[f'has_{condition}'] = claims_features[f'has_{condition}'].fillna(0)

        # Comorbidity score (number of unique diagnosis codes)
        unique_diagnoses = self.claims.groupby('member_id')['icd_code'].nunique().reset_index()
        unique_diagnoses.columns = ['member_id', 'comorbidity_score']
        claims_features = claims_features.merge(unique_diagnoses, on='member_id', how='left')

        # Drop timestamp columns
        claims_features = claims_features.drop(['first_diagnosis', 'last_diagnosis'], axis=1)

        return claims_features
