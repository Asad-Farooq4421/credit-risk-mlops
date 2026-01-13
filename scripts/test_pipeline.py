#!/usr/bin/env python
"""
Test script to verify the complete ML pipeline.
Run this script to test all components of the credit risk prediction system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

def test_data_loading():
    """Test data loading functionality."""
    print("=" * 60)
    print("TEST 1: Data Loading")
    print("=" * 60)

    try:
        # Load configuration
        config_path = Path("config/config.yaml")
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            config = yaml.safe_load(f)

        # Load German Credit data
        data_path = Path(config['data']['raw_path']) / 'german_credit.csv'
        df = pd.read_csv(data_path)

        print(f"✓ Configuration loaded from: {config_path}")
        print(f"✓ Data loaded from: {data_path}")
        print(f"✓ Data shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)[:5]}...")  # Show first 5 columns
        print(f"✓ Target variable '{config['model']['target']}': {df[config['model']['target']].unique()}")

        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering pipeline."""
    print("\n" + "=" * 60)
    print("TEST 2: Feature Engineering")
    print("=" * 60)

    try:
        from src.features.build_features import FeatureEngineer

        # Load sample data
        config_path = Path("config/config.yaml")
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            config = yaml.safe_load(f)

        data_path = Path(config['data']['raw_path']) / 'german_credit.csv'
        df = pd.read_csv(data_path).head(100)  # Use first 100 rows for testing

        # Initialize feature engineer
        engineer = FeatureEngineer()

        # Identify feature types
        num_feats, cat_feats = engineer.identify_feature_types(df)

        print(f"✓ FeatureEngineer initialized")
        print(f"✓ Numerical features ({len(num_feats)}): {num_feats[:3]}...")  # Show first 3
        print(f"✓ Categorical features ({len(cat_feats)}): {cat_feats[:3]}...")  # Show first 3

        # Prepare training data
        X, y = engineer.prepare_training_data(df)

        print(f"✓ Training data prepared")
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")
        print(f"✓ Feature names: {X.columns.tolist()[:5]}...")  # Show first 5

        return True
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_validation():
    """Test data validation functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Data Validation")
    print("=" * 60)

    try:
        from src.data.validation import DataValidator

        # Load sample data
        config_path = Path("config/config.yaml")
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            config = yaml.safe_load(f)

        data_path = Path(config['data']['raw_path']) / 'german_credit.csv'
        df = pd.read_csv(data_path).head(50)  # Use first 50 rows

        # Initialize validator
        validator = DataValidator("german_credit")

        # Generate report
        report = validator.generate_report(df)

        print(f"✓ DataValidator initialized")
        print(f"✓ Validation status: {report['validation']['status']}")
        print(f"✓ Data quality checks completed")
        print(f"✓ Missing values: {report['data_quality']['missing_data']['total_missing']}")
        print(f"✓ Duplicates: {report['data_quality']['duplicates']['total_duplicates']}")
        print(f"✓ Recommendations: {report['recommendations'][0]}")

        return True
    except Exception as e:
        print(f"✗ Data validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mlflow_setup():
    """Test MLflow setup."""
    print("\n" + "=" * 60)
    print("TEST 4: MLflow Setup")
    print("=" * 60)

    try:
        import mlflow

        # Load configuration
        config_path = Path("config/config.yaml")
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            config = yaml.safe_load(f)

        # Setup MLflow
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])

        print(f"✓ MLflow tracking URI: {mlflow.get_tracking_uri()}")

        # Get experiment info
        experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
        if experiment:
            print(f"✓ MLflow experiment: {experiment.name}")
        else:
            print(f"✓ MLflow experiment created: {config['mlflow']['experiment_name']}")

        print(f"✓ MLflow setup successful")

        # Test a simple MLflow run
        with mlflow.start_run(run_name="test_run"):
            mlflow.log_param("test_param", 42)
            mlflow.log_metric("test_metric", 0.95)
            print(f"✓ Test MLflow run created")

        return True
    except Exception as e:
        print(f"✗ MLflow setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_training():
    """Test model training with small dataset."""
    print("\n" + "=" * 60)
    print("TEST 5: Model Training (Quick Test)")
    print("=" * 60)

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        # Load and prepare small dataset
        config_path = Path("config/config.yaml")
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            config = yaml.safe_load(f)

        data_path = Path(config['data']['raw_path']) / 'german_credit.csv'
        df = pd.read_csv(data_path).head(200)  # Small subset for quick test

        # Simple preprocessing
        X = df.drop(columns=[config['model']['target']])
        y = df[config['model']['target']]

        # Simple encoding for categorical columns
        X = pd.get_dummies(X, drop_first=True)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"✓ Data prepared: {X.shape}")
        print(f"✓ Model trained: {model.__class__.__name__}")
        print(f"✓ Test accuracy: {accuracy:.4f}")
        print(f"✓ Feature importance calculated: {len(model.feature_importances_)} features")

        return True
    except Exception as e:
        print(f"✗ Model training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_directory_structure():
    """Verify project directory structure."""
    print("\n" + "=" * 60)
    print("TEST 6: Directory Structure")
    print("=" * 60)

    required_dirs = [
        "data/raw",
        "data/processed",
        "src/data",
        "src/features",
        "src/models",
        "src/utils",
        "notebooks",
        "config",
        "tests",
        "scripts"
    ]

    required_files = [
        "config/config.yaml",
        "src/data/make_dataset.py",
        "src/data/validation.py",
        "src/features/build_features.py",
        "src/models/train.py",
        "src/utils/logger.py",
        "requirements.txt",
        "README.md"
    ]

    all_passed = True

    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")
            all_passed = False

    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ File exists: {file_path}")
        else:
            print(f"✗ Missing file: {file_path}")
            all_passed = False

    return all_passed

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CREDIT RISK ML PIPELINE - COMPREHENSIVE TEST")
    print("=" * 60)

    test_results = []

    # Run tests
    test_results.append(("Data Loading", test_data_loading()))
    test_results.append(("Feature Engineering", test_feature_engineering()))
    test_results.append(("Data Validation", test_data_validation()))
    test_results.append(("MLflow Setup", test_mlflow_setup()))
    test_results.append(("Model Training", test_model_training()))
    test_results.append(("Directory Structure", test_directory_structure()))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)

    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 60)

    if passed_tests == total_tests:
        print("\n✅ ALL TESTS PASSED! Pipeline is ready.")
        print("\nNext steps:")
        print("1. Run the EDA notebook: notebooks/01_german_credit_eda.ipynb")
        print("2. Train full model: python src/models/train.py")
        print("3. Start API: uvicorn api.app:app --reload")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before proceeding.")

    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
