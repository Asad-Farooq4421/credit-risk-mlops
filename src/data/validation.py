import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataValidator:
    """
    Data validation class for credit risk datasets.
    Uses custom validation logic.
    """

    def __init__(self, dataset_name: str = "german_credit"):
        """
        Initialize validator for specific dataset.

        Args:
            dataset_name: Name of dataset ('german_credit' or 'lending_club')
        """
        self.dataset_name = dataset_name
        self.validation_report = {}

        # Define valid values for German Credit dataset
        self.german_credit_validation = {
            "status": ['A11', 'A12', 'A13', 'A14'],
            "credit_history": ['A30', 'A31', 'A32', 'A33', 'A34'],
            "purpose": ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410'],
            "savings": ['A61', 'A62', 'A63', 'A64', 'A65'],
            "employment_duration": ['A71', 'A72', 'A73', 'A74', 'A75'],
            "personal_status_sex": ['A91', 'A92', 'A93', 'A94', 'A95'],
            "other_debtors": ['A101', 'A102', 'A103'],
            "property": ['A121', 'A122', 'A123', 'A124'],
            "other_installment_plans": ['A141', 'A142', 'A143'],
            "housing": ['A151', 'A152', 'A153'],
            "job": ['A171', 'A172', 'A173', 'A174'],
            "telephone": ['A191', 'A192'],
            "foreign_worker": ['A201', 'A202'],
            "credit_risk": [1, 2]
        }

        # Numerical ranges for German Credit
        self.german_credit_ranges = {
            "duration": (0, 100),
            "credit_amount": (0, 20000),
            "installment_rate": (0, 5),
            "present_residence": (0, 5),
            "age": (18, 100),
            "number_credits": (0, 5),
            "people_liable": (0, 3)
        }

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate dataset against schema.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, validation_report)
        """
        errors = []

        # Validate categorical columns
        for col, valid_values in self.german_credit_validation.items():
            if col in df.columns:
                invalid_values = df[~df[col].isin(valid_values)][col].unique()
                if len(invalid_values) > 0:
                    errors.append({
                        "column": col,
                        "check": "valid_values",
                        "error": f"Invalid values: {invalid_values[:5]}",
                        "count": len(df[~df[col].isin(valid_values)])
                    })

        # Validate numerical ranges
        for col, (min_val, max_val) in self.german_credit_ranges.items():
            if col in df.columns:
                out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                if len(out_of_range) > 0:
                    errors.append({
                        "column": col,
                        "check": "value_range",
                        "error": f"Values outside range ({min_val}, {max_val})",
                        "count": len(out_of_range)
                    })

        # Check for missing values
        missing_cols = df.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        for col in missing_cols.index:
            errors.append({
                "column": col,
                "check": "missing_values",
                "error": f"Missing values found",
                "count": int(missing_cols[col])
            })

        self.validation_report = {
            "status": "PASS" if len(errors) == 0 else "FAIL",
            "errors": errors,
            "warnings": [],
            "summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "missing_values": df.isnull().sum().sum(),
                "duplicate_rows": df.duplicated().sum()
            }
        }

        return len(errors) == 0, self.validation_report

    def check_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive data quality checks.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary with data quality metrics
        """
        quality_report = {
            "basic_stats": {
                "shape": df.shape,
                "dtypes": df.dtypes.to_dict(),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
            },
            "missing_data": {
                "total_missing": df.isnull().sum().sum(),
                "missing_by_column": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            },
            "duplicates": {
                "total_duplicates": df.duplicated().sum(),
                "duplicate_percentage": (df.duplicated().sum() / len(df)) * 100
            },
            "data_types": {
                "numerical_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
                "datetime_columns": df.select_dtypes(include=['datetime']).columns.tolist()
            },
            "outliers": self._detect_outliers(df),
            "cardinality": self._check_cardinality(df)
        }

        return quality_report

    def _detect_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> Dict:
        """Detect outliers using IQR method."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outliers_report = {}

        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_report[col] = {
                "count": len(outliers),
                "percentage": (len(outliers) / len(df)) * 100,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }

        return outliers_report

    def _check_cardinality(self, df: pd.DataFrame) -> Dict:
        """Check cardinality of categorical columns."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        cardinality_report = {}

        for col in categorical_cols:
            unique_values = df[col].nunique()
            cardinality_report[col] = {
                "unique_count": unique_values,
                "high_cardinality": unique_values > 50,  # Threshold for high cardinality
                "sample_values": df[col].unique()[:5].tolist() if unique_values > 5 else df[col].unique().tolist()
            }

        return cardinality_report

    def generate_report(self, df: pd.DataFrame, save_path: Optional[str] = None) -> Dict:
        """
        Generate comprehensive validation report.

        Args:
            df: DataFrame to validate
            save_path: Path to save report (optional)

        Returns:
            Complete validation report
        """
        # Validate with schema
        is_valid, validation_result = self.validate_data(df)

        # Check data quality
        quality_report = self.check_data_quality(df)

        # Combine reports
        complete_report = {
            "dataset": self.dataset_name,
            "validation": validation_result,
            "data_quality": quality_report,
            "recommendations": self._generate_recommendations(validation_result, quality_report)
        }

        # Save report if path provided
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(complete_report, f, indent=2, default=str)
            print(f"Report saved to: {save_path}")

        return complete_report

    def _generate_recommendations(self, validation_result: Dict, quality_report: Dict) -> List[str]:
        """Generate recommendations based on validation and quality checks."""
        recommendations = []

        # Check for validation errors
        if validation_result["status"] == "FAIL":
            recommendations.append("Fix schema validation errors before proceeding")

        # Check for missing values
        if quality_report["missing_data"]["total_missing"] > 0:
            recommendations.append("Handle missing values using imputation or removal")

        # Check for duplicates
        if quality_report["duplicates"]["total_duplicates"] > 0:
            recommendations.append("Remove duplicate rows from dataset")

        # Check for high cardinality
        for col, stats in quality_report["cardinality"].items():
            if stats["high_cardinality"]:
                recommendations.append(f"Consider encoding or reducing cardinality for column: {col}")

        # Check for outliers
        for col, stats in quality_report["outliers"].items():
            if stats["count"] > 0:
                recommendations.append(f"Investigate outliers in column: {col}")

        return recommendations if recommendations else ["Data quality is good. Proceed with modeling."]

# Example usage
if __name__ == "__main__":
    # Test with sample data
    import pandas as pd

    # Create sample data
    sample_data = pd.DataFrame({
        "status": ["A11", "A12", "A13"],
        "duration": [6, 48, 12],
        "credit_history": ["A30", "A31", "A32"],
        "credit_risk": [1, 2, 1]
    })

    # Initialize validator
    validator = DataValidator("german_credit")

    # Generate report
    report = validator.generate_report(sample_data)

    print("Validation Report:")
    print(f"Status: {report['validation']['status']}")
    print(f"Errors: {len(report['validation']['errors'])}")
    print(f"Recommendations: {report['recommendations']}")
