[README.md](https://github.com/user-attachments/files/21835641/README.md)
# WellCo Churn Prediction

A machine learning system to predict customer churn for WellCo health platform and generate prioritized outreach recommendations with optimal threshold selection.

## Features

- **Multi-source feature engineering**: Combines web visits, app usage, claims data, and temporal features
- **Multiple ML models**: Compares Logistic Regression, Random Forest, Gradient Boosting, and MLP
- **Optimal threshold selection**: Automatically finds the best classification threshold for business metrics
- **Cost-benefit optimization**: Determines optimal outreach size based on ROI analysis
- **Comprehensive evaluation**: Generates detailed reports, visualizations, and performance comparisons

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd wellco-churn-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data**: Place these CSV files in the project directory:
   - `churn_labels.csv` - Member IDs, churn labels, signup dates, outreach flags
   - `web_visits.zip` - Web visit logs with URLs, titles, descriptions, timestamps
   - `app_usage.csv` - Mobile app usage events and timestamps
   - `claims.csv` - Medical claims with ICD codes and diagnosis dates

4. **Run analysis**:
   ```bash
   python main.py
   ```

## Output Files

- `wellco_outreach_recommendations.csv` - Top members ranked by churn probability
- `wellco_churn_model.pkl` - Trained model with optimal threshold
- `wellco_analysis_results.png` - Performance visualizations and threshold analysis
- `wellco_churn_analysis.log` - Complete analysis log

## Data Requirements

### Required CSV Columns

**churn_labels.csv**:
- `member_id`, `churn`, `signup_date`, `outreach`

**web_visits.csv** (in zip file):
- `member_id`, `url`, `title`, `description`, `timestamp`

**app_usage.csv**:
- `member_id`, `event_type`, `timestamp`

**claims.csv**:
- `member_id`, `icd_code`, `diagnosis_date`

## Key Components

- **Feature Engineering** (`features.py`): Creates engagement, temporal, and clinical features
- **Model Training** (`models.py`): Trains multiple models with optimal threshold selection
- **Evaluation** (`evaluation.py`): Generates comprehensive performance analysis
- **Configuration** (`config.py`): Centralized settings for model parameters and business rules

## Model Performance

The system automatically selects the best-performing model based on AUC score and optimizes the classification threshold using configurable metrics (F1, precision, recall, or balanced accuracy).

## Business Focus Areas

Analyzes engagement across WellCo's key content categories:
- Nutrition and diet
- Exercise and fitness
- Sleep and rest
- Mental health and wellness

Prioritizes members with conditions: diabetes, hypertension, and dietary counseling needs.

## Configuration

Modify `config.py` to adjust:
- Test/train split ratio
- Cross-validation folds
- Threshold optimization metric
- Recent activity window
- Business constants and ICD codes
