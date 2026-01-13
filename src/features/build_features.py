import warnings
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering pipeline for credit risk datasets.
    Handles preprocessing, feature creation, and transformation.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize feature engineer with configuration.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.numerical_features = []
        self.categorical_features = []
        self.target_column = self.config['model']['target']
        self.preprocessor = None
        self.feature_names = []

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        with open(config_file, 'r', encoding='utf-8-sig') as f:
            config = yaml.safe_load(f)
        return config

    def identify_feature_types(self, df: pd.DataFrame) -> Tuple[List, List]:
        """
        Identify numerical and categorical features.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (numerical_features, categorical_features)
        """
        # Exclude target column
        features = [col for col in df.columns if col != self.target_column]

        # Identify feature types
        self.numerical_features = df[features].select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()

        self.categorical_features = df[features].select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

        return self.numerical_features, self.categorical_features

    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline for features.

        Returns:
            ColumnTransformer with preprocessing steps
        """
        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Create column transformer
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, self.numerical_features),
            ('cat', categorical_pipeline, self.categorical_features)
        ])

        return self.preprocessor

    def engineer_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Perform feature engineering on dataset.

        Args:
            df: Input DataFrame
            is_training: Whether this is training data

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # 1. Handle missing values
        df = self._handle_missing_values(df)

        # 2. Create new features
        df = self._create_interaction_features(df)
        df = self._create_polynomial_features(df)
        df = self._create_statistical_features(df)
        df = self._create_domain_specific_features(df)

        # 3. Encode categorical variables
        if is_training:
            self._fit_encoders(df)

        df = self._encode_categorical_features(df)

        # 4. Scale numerical features
        if is_training:
            self._fit_scalers(df)

        df = self._scale_numerical_features(df)

        # 5. Feature selection (optional)
        df = self._select_features(df)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # For numerical columns, fill with median
        for col in self.numerical_features:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # For categorical columns, fill with mode
        for col in self.categorical_features:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables."""
        # Example interaction features for credit risk
        if 'credit_amount' in df.columns and 'duration' in df.columns:
            df['amount_per_month'] = df['credit_amount'] / df['duration']

        if 'annual_inc' in df.columns and 'loan_amnt' in df.columns:
            df['debt_to_income_ratio'] = df['loan_amnt'] / df['annual_inc']

        if 'age' in df.columns and 'credit_amount' in df.columns:
            df['age_amount_ratio'] = df['credit_amount'] / df['age']

        return df

    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features for numerical columns."""
        # Square of important numerical features
        important_numerical = ['credit_amount', 'duration', 'age', 'annual_inc', 'loan_amnt']

        for col in important_numerical:
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_sqrt'] = np.sqrt(df[col].abs())

        return df

    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        # Create bins for numerical features
        if 'credit_amount' in df.columns:
            df['credit_amount_bin'] = pd.qcut(df['credit_amount'], q=5, labels=False)

        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'],
                                     bins=[0, 25, 35, 50, 65, 100],
                                     labels=['18-25', '26-35', '36-50', '51-65', '65+'])

        if 'annual_inc' in df.columns:
            df['income_group'] = pd.qcut(df['annual_inc'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

        return df

    def _create_domain_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for credit risk."""
        # Credit utilization ratio (if applicable)
        if all(col in df.columns for col in ['credit_amount', 'savings']):
            # Map savings to numerical values
            savings_map = {'A61': 0, 'A62': 100, 'A63': 500, 'A64': 1000, 'A65': 5000}
            if df['savings'].isin(savings_map.keys()).all():
                df['savings_numeric'] = df['savings'].map(savings_map)
                df['credit_utilization'] = df['credit_amount'] / (df['savings_numeric'] + 1)

        # Employment stability score
        if 'employment_duration' in df.columns:
            employment_map = {
                'A71': 0,  # unemployed
                'A72': 1,  # <1 year
                'A73': 2,  # 1-4 years
                'A74': 3,  # 4-7 years
                'A75': 4   # >=7 years
            }
            if df['employment_duration'].isin(employment_map.keys()).all():
                df['employment_stability'] = df['employment_duration'].map(employment_map)

        # Risk score based on multiple factors
        risk_factors = []
        if 'credit_history' in df.columns:
            history_map = {'A30': 0, 'A31': 1, 'A32': 2, 'A33': 3, 'A34': 4}
            if df['credit_history'].isin(history_map.keys()).all():
                df['history_score'] = df['credit_history'].map(history_map)
                risk_factors.append('history_score')

        if 'property' in df.columns:
            property_map = {'A121': 0, 'A122': 1, 'A123': 2, 'A124': 3}
            if df['property'].isin(property_map.keys()).all():
                df['property_score'] = df['property'].map(property_map)
                risk_factors.append('property_score')

        # Create composite risk score
        if risk_factors:
            df['composite_risk_score'] = df[risk_factors].sum(axis=1)

        return df

    def _fit_encoders(self, df: pd.DataFrame):
        """Fit encoders for categorical features."""
        self.label_encoders = {}
        self.onehot_encoders = {}

        for col in self.categorical_features:
            if col in df.columns:
                # For binary categorical, use LabelEncoder
                if df[col].nunique() == 2:
                    le = LabelEncoder()
                    le.fit(df[col])
                    self.label_encoders[col] = le
                # For multi-class categorical, use OneHotEncoder
                else:
                    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    ohe.fit(df[[col]])
                    self.onehot_encoders[col] = ohe

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        df_encoded = df.copy()

        # Apply LabelEncoder for binary features
        for col, encoder in self.label_encoders.items():
            if col in df_encoded.columns:
                df_encoded[col] = encoder.transform(df_encoded[col])

        # Apply OneHotEncoder for multi-class features
        for col, encoder in self.onehot_encoders.items():
            if col in df_encoded.columns:
                encoded_data = encoder.transform(df_encoded[[col]])
                encoded_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df_encoded.index)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), encoded_df], axis=1)

        return df_encoded

    def _fit_scalers(self, df: pd.DataFrame):
        """Fit scalers for numerical features."""
        self.scalers = {}

        for col in self.numerical_features:
            if col in df.columns:
                scaler = StandardScaler()
                scaler.fit(df[[col]])
                self.scalers[col] = scaler

    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        df_scaled = df.copy()

        for col, scaler in self.scalers.items():
            if col in df_scaled.columns:
                df_scaled[col] = scaler.transform(df_scaled[[col]])

        return df_scaled

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select important features (placeholder for actual feature selection)."""
        # This is a placeholder. In practice, you would use:
        # 1. Correlation analysis
        # 2. Feature importance from tree-based models
        # 3. Recursive feature elimination
        # 4. Mutual information

        # For now, return all features
        return df

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with features and target.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (features, target)
        """
        # Identify feature types
        self.identify_feature_types(df)

        # Engineer features
        X = self.engineer_features(df, is_training=True)

        # Extract target
        y = df[self.target_column]

        # Store feature names
        self.feature_names = X.columns.tolist()

        return X, y

    def prepare_test_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare test data using fitted transformers.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        # Engineer features using fitted transformers
        X = self.engineer_features(df, is_training=False)

        # Ensure same columns as training
        missing_cols = set(self.feature_names) - set(X.columns)
        for col in missing_cols:
            X[col] = 0

        extra_cols = set(X.columns) - set(self.feature_names)
        X = X[self.feature_names]

        return X

    def save_artifacts(self, save_path: str = "artifacts/feature_engineering"):
        """
        Save feature engineering artifacts.

        Args:
            save_path: Path to save artifacts
        """
        artifacts_dir = Path(save_path)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'feature_names': self.feature_names,
            'target_column': self.target_column,
            'label_encoders': self.label_encoders if hasattr(self, 'label_encoders') else {},
            'onehot_encoders': self.onehot_encoders if hasattr(self, 'onehot_encoders') else {},
            'scalers': self.scalers if hasattr(self, 'scalers') else {}
        }

        joblib.dump(artifacts, artifacts_dir / 'feature_artifacts.pkl')
        print(f"Feature engineering artifacts saved to: {artifacts_dir / 'feature_artifacts.pkl'}")

    def load_artifacts(self, load_path: str = "artifacts/feature_engineering"):
        """
        Load feature engineering artifacts.

        Args:
            load_path: Path to load artifacts from
        """
        artifacts_path = Path(load_path) / 'feature_artifacts.pkl'

        if artifacts_path.exists():
            artifacts = joblib.load(artifacts_path)
            self.numerical_features = artifacts['numerical_features']
            self.categorical_features = artifacts['categorical_features']
            self.feature_names = artifacts['feature_names']
            self.target_column = artifacts['target_column']

            if 'label_encoders' in artifacts:
                self.label_encoders = artifacts['label_encoders']
            if 'onehot_encoders' in artifacts:
                self.onehot_encoders = artifacts['onehot_encoders']
            if 'scalers' in artifacts:
                self.scalers = artifacts['scalers']

            print(f"Feature engineering artifacts loaded from: {artifacts_path}")
        else:
            print(f"No artifacts found at: {artifacts_path}")

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({
        'status': ['A11', 'A12', 'A13', 'A14', 'A11'],
        'duration': [6, 48, 12, 42, 24],
        'credit_amount': [1169, 5951, 2096, 7882, 4870],
        'age': [25, 35, 45, 55, 65],
        'savings': ['A61', 'A62', 'A63', 'A64', 'A65'],
        'employment_duration': ['A71', 'A72', 'A73', 'A74', 'A75'],
        'credit_history': ['A30', 'A31', 'A32', 'A33', 'A34'],
        'property': ['A121', 'A122', 'A123', 'A124', 'A121'],
        'credit_risk': [1, 2, 1, 1, 2]
    })

    # Initialize feature engineer
    engineer = FeatureEngineer()

    # Prepare training data
    X_train, y_train = engineer.prepare_training_data(sample_data)

    print("Feature Engineering Results:")
    print(f"Original shape: {sample_data.shape}")
    print(f"Engineered features shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    print(f"\\nEngineered feature names:")
    print(X_train.columns.tolist())
    print(f"\\nFirst few rows of engineered features:")
    print(X_train.head())

    # Save artifacts
    engineer.save_artifacts()
