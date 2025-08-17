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

    def create_web_features(self):
        """Create features from web visit data"""
        if self.web_visits.empty:
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