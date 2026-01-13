import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import io

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path("config/config.yaml")
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)
    return config

def download_german_credit_data():
    """Download German Credit Dataset from UCI"""
    print("Downloading German Credit Dataset...")
    config = load_config()

    # URL for German Credit Data
    url = config['data']['german_credit_url']

    # Column names as per UCI documentation
    column_names = [
        'status', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings', 'employment_duration', 'installment_rate', 'personal_status_sex',
        'other_debtors', 'present_residence', 'property', 'age', 'other_installment_plans',
        'housing', 'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
        'credit_risk'
    ]

    try:
        # Download data
        response = requests.get(url)
        response.raise_for_status()

        # Read data
        data = pd.read_csv(io.StringIO(response.text), sep=' ', header=None, names=column_names)

        # Save to raw data folder
        raw_path = Path(config['data']['raw_path'])
        raw_path.mkdir(parents=True, exist_ok=True)

        file_path = raw_path / 'german_credit.csv'
        data.to_csv(file_path, index=False)

        print(f"✓ German Credit Dataset downloaded successfully: {file_path}")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {list(data.columns)}")

        # Display basic info
        print("\nDataset Preview:")
        print(data.head())
        print(f"\nTarget variable distribution (credit_risk):")
        print(data['credit_risk'].value_counts())

        return data

    except Exception as e:
        print(f"✗ Error downloading German Credit Dataset: {e}")
        return None

def create_sample_lending_club_data():
    """Create a sample Lending Club dataset for demonstration"""
    print("\nCreating sample Lending Club data...")
    config = load_config()

    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000

    sample_data = pd.DataFrame({
        'loan_amnt': np.random.randint(5000, 35000, n_samples),
        'term': np.random.choice(['36 months', '60 months'], n_samples),
        'int_rate': np.random.uniform(5, 30, n_samples),
        'installment': np.random.uniform(100, 1000, n_samples),
        'grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_samples),
        'emp_length': np.random.choice(['<1 year', '1 year', '2-4 years', '5-9 years', '10+ years'], n_samples),
        'home_ownership': np.random.choice(['RENT', 'MORTGAGE', 'OWN', 'OTHER'], n_samples),
        'annual_inc': np.random.uniform(20000, 150000, n_samples),
        'verification_status': np.random.choice(['Verified', 'Source Verified', 'Not Verified'], n_samples),
        'purpose': np.random.choice(['credit_card', 'car', 'small_business', 'home_improvement'], n_samples),
        'dti': np.random.uniform(0, 40, n_samples),
        'delinq_2yrs': np.random.randint(0, 5, n_samples),
        'fico_range_low': np.random.randint(600, 850, n_samples),
        'fico_range_high': np.random.randint(650, 900, n_samples),
        'loan_status': np.random.choice(['Fully Paid', 'Charged Off', 'Current'], n_samples, p=[0.8, 0.15, 0.05]),
        'credit_risk': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 0=Good, 1=Bad
    })

    # Save to raw data folder
    raw_path = Path(config['data']['raw_path'])
    raw_path.mkdir(parents=True, exist_ok=True)

    file_path = raw_path / 'lending_club_sample.csv'
    sample_data.to_csv(file_path, index=False)

    print(f"✓ Sample Lending Club data created: {file_path}")
    print(f"  Shape: {sample_data.shape}")
    print(f"  Columns: {list(sample_data.columns)}")

    return sample_data

def main():
    """Main function to download all datasets"""
    print("=" * 60)
    print("DATASET ACQUISITION")
    print("=" * 60)

    # Download German Credit Dataset
    german_data = download_german_credit_data()

    # Create sample Lending Club data
    lending_data = create_sample_lending_club_data()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if german_data is not None:
        print(f"German Credit Dataset: {german_data.shape[0]} rows, {german_data.shape[1]} columns")
    if lending_data is not None:
        print(f"Lending Club Sample: {lending_data.shape[0]} rows, {lending_data.shape[1]} columns")

    print("\n✓ All datasets have been prepared in data/raw/")
    print("  Note: For full Lending Club dataset, download from Kaggle manually")
    print("  Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club")

if __name__ == "__main__":
    main()
