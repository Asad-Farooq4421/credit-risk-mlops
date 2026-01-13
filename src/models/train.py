import warnings

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
from pathlib import Path
import yaml
import json
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.features.build_features import FeatureEngineer
from src.utils.logger import setup_logger

class ModelTrainer:
    """
    Model training and evaluation pipeline for credit risk prediction.
    Supports multiple algorithms, hyperparameter tuning, and MLflow tracking.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize model trainer with configuration.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = setup_logger("model_trainer")
        self.feature_engineer = FeatureEngineer(config_path)
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
        self.mlflow_experiment_name = self.config['mlflow']['experiment_name']

        # Setup MLflow
        self._setup_mlflow()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        with open(config_file, 'r', encoding='utf-8-sig') as f:
            config = yaml.safe_load(f)
        return config

    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.mlflow_experiment_name)
        self.logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        self.logger.info(f"MLflow experiment: {self.mlflow_experiment_name}")

    def load_and_prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare data for training.

        Args:
            data_path: Path to data file

        Returns:
            Tuple of (features, target)
        """
        self.logger.info(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)

        # Prepare features and target
        X, y = self.feature_engineer.prepare_training_data(df)

        self.logger.info(f"Data shape: {X.shape}")
        self.logger.info(f"Target distribution:\\n{y.value_counts()}")

        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """
        Split data into train, validation, and test sets.

        Args:
            X: Features
            y: Target

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        test_size = self.config['model']['test_size']
        random_state = self.config['model']['random_state']

        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=test_size, random_state=random_state, stratify=y_train_val
        )

        self.logger.info(f"Train set: {X_train.shape}")
        self.logger.info(f"Validation set: {X_val.shape}")
        self.logger.info(f"Test set: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_model_configurations(self) -> Dict:
        """
        Get model configurations for training.

        Returns:
            Dictionary of model configurations
        """
        model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'xgboost': {
                'model': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                }
            }
        }

        return model_configs

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """
        Train multiple models with hyperparameter tuning.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data

        Returns:
            Dictionary of trained models
        """
        self.logger.info("Starting model training with hyperparameter tuning...")

        model_configs = self.get_model_configurations()
        cv_folds = self.config['model']['cv_folds']
        random_state = self.config['model']['random_state']

        for model_name, config in model_configs.items():
            self.logger.info(f"\\nTraining {model_name}...")

            with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_param("model_name", model_name)
                mlflow.log_params({"cv_folds": cv_folds, "random_state": random_state})

                # Hyperparameter tuning with GridSearchCV
                grid_search = GridSearchCV(
                    estimator=config['model'],
                    param_grid=config['params'],
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )

                # Train model
                grid_search.fit(X_train, y_train)

                # Get best model
                best_model = grid_search.best_estimator_
                self.models[model_name] = best_model

                # Log best parameters
                mlflow.log_params(grid_search.best_params_)

                # Evaluate on validation set
                val_metrics = self.evaluate_model(best_model, X_val, y_val, "validation")

                # Log metrics to MLflow
                for metric_name, metric_value in val_metrics.items():
                    mlflow.log_metric(f"val_{metric_name}", metric_value)

                # Log model
                mlflow.sklearn.log_model(best_model, model_name)

                self.logger.info(f"{model_name} - Best params: {grid_search.best_params_}")
                self.logger.info(f"{model_name} - Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")

        return self.models

    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series,
                      dataset_name: str = "test") -> Dict:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            X: Features
            y: True labels
            dataset_name: Name of dataset for logging

        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }

        # Add ROC-AUC if probability predictions available
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)

        # Store confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        metrics['classification_report'] = report

        # Log metrics
        self.logger.info(f"{dataset_name.capitalize()} metrics for {model.__class__.__name__}:")
        for metric_name, metric_value in metrics.items():
            if metric_name not in ['confusion_matrix', 'classification_report']:
                self.logger.info(f"  {metric_name}: {metric_value:.4f}")

        return metrics

    def select_best_model(self, X_val: pd.DataFrame, y_val: pd.Series):
        """
        Select the best performing model based on validation metrics.

        Args:
            X_val, y_val: Validation data
        """
        best_score = -1
        self.best_model = None
        self.best_model_name = None

        self.logger.info("\\nSelecting best model...")

        for model_name, model in self.models.items():
            # Evaluate model
            metrics = self.evaluate_model(model, X_val, y_val, "validation")
            score = metrics.get('roc_auc', metrics['f1'])

            if score > best_score:
                best_score = score
                self.best_model = model
                self.best_model_name = model_name

        self.logger.info(f"Best model: {self.best_model_name} with score: {best_score:.4f}")

    def create_visualizations(self, model: Any, X_test: pd.DataFrame,
                            y_test: pd.Series, save_path: Optional[str] = None):
        """
        Create model evaluation visualizations.

        Args:
            model: Trained model
            X_test, y_test: Test data
            save_path: Path to save visualizations
        """
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if save_path:
            plt.savefig(save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. ROC Curve (if probability available)
        if y_pred_proba is not None:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            if save_path:
                plt.savefig(save_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 3. Feature Importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 8))
            feature_importance = pd.DataFrame({
                'feature': self.feature_engineer.feature_names[:len(model.feature_importances_)],
                'importance': model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False).head(20)

            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title('Top 20 Feature Importances')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()

    def save_model(self, model: Any, model_name: str, save_path: str = "models"):
        """
        Save trained model to disk.

        Args:
            model: Trained model
            model_name: Name of the model
            save_path: Path to save the model
        """
        models_dir = Path(save_path)
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = models_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)

        # Save feature engineering artifacts
        self.feature_engineer.save_artifacts()

        self.logger.info(f"Model saved to: {model_path}")
        return model_path

    def generate_report(self, X_test: pd.DataFrame, y_test: pd.Series,
                       save_path: Optional[str] = None) -> Dict:
        """
        Generate comprehensive model evaluation report.

        Args:
            X_test, y_test: Test data
            save_path: Path to save report

        Returns:
            Dictionary with complete report
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Run select_best_model first.")

        # Evaluate best model on test set
        test_metrics = self.evaluate_model(self.best_model, X_test, y_test, "test")

        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.best_model_name,
            'model_type': type(self.best_model).__name__,
            'test_metrics': test_metrics,
            'feature_importance': self._get_feature_importance() if hasattr(self.best_model, 'feature_importances_') else None,
            'training_config': self.config['model'],
            'dataset_info': {
                'test_samples': len(X_test),
                'feature_count': X_test.shape[1],
                'class_distribution': y_test.value_counts().to_dict()
            }
        }

        # Save report if path provided
        if save_path:
            report_path = Path(save_path) / f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Model report saved to: {report_path}")

        return report

    def _get_feature_importance(self) -> Dict:
        """Get feature importance from model."""
        if hasattr(self.best_model, 'feature_importances_'):
            importance_dict = dict(zip(
                self.feature_engineer.feature_names[:len(self.best_model.feature_importances_)],
                self.best_model.feature_importances_.tolist()
            ))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return {}

    def run_pipeline(self, data_path: str):
        """
        Run complete training pipeline.

        Args:
            data_path: Path to data file
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING MODEL TRAINING PIPELINE")
        self.logger.info("=" * 60)

        try:
            # 1. Load and prepare data
            X, y = self.load_and_prepare_data(data_path)

            # 2. Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

            # 3. Train models
            self.train_models(X_train, y_train, X_val, y_val)

            # 4. Select best model
            self.select_best_model(X_val, y_val)

            # 5. Evaluate on test set
            test_metrics = self.evaluate_model(self.best_model, X_test, y_test, "test")

            # 6. Create visualizations
            self.create_visualizations(self.best_model, X_test, y_test, save_path="reports/visualizations")

            # 7. Save best model
            model_path = self.save_model(self.best_model, self.best_model_name)

            # 8. Generate report
            report = self.generate_report(X_test, y_test, save_path="reports")

            # 9. Log final results
            with mlflow.start_run(run_name=f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("best_model", self.best_model_name)
                mlflow.log_params(self.best_model.get_params())

                for metric_name, metric_value in test_metrics.items():
                    if metric_name not in ['confusion_matrix', 'classification_report']:
                        mlflow.log_metric(f"test_{metric_name}", metric_value)

                mlflow.log_artifact(model_path)
                mlflow.log_artifact("reports/")

                # Log feature importance plot if available
                if hasattr(self.best_model, 'feature_importances_'):
                    importance_plot = "reports/visualizations/feature_importance.png"
                    if Path(importance_plot).exists():
                        mlflow.log_artifact(importance_plot)

            self.logger.info("\\n" + "=" * 60)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)
            self.logger.info(f"Best model: {self.best_model_name}")
            self.logger.info(f"Test ROC-AUC: {test_metrics.get('roc_auc', 'N/A'):.4f}")
            self.logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            self.logger.info(f"Model saved to: {model_path}")

        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {e}")
            raise

# Main execution
if __name__ == "__main__":
    # Initialize trainer
    trainer = ModelTrainer()

    # Run pipeline
    data_path = "data/raw/german_credit.csv"
    trainer.run_pipeline(data_path)
