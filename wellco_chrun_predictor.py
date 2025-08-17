import pandas as pd

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
