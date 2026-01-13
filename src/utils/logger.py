import logging
import sys
from pathlib import Path
import yaml

def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    with open(config_file, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)
    return config

class ProjectPaths:
    """Centralized path management for the project."""

    def __init__(self):
        self.config = load_config()
        self.base_dir = Path(".")

    @property
    def raw_data_dir(self) -> Path:
        return self.base_dir / self.config['data']['raw_path']

    @property
    def processed_data_dir(self) -> Path:
        return self.base_dir / self.config['data']['processed_path']

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def mlflow_dir(self) -> Path:
        return self.base_dir / self.config['mlflow']['tracking_uri']

    def create_directories(self):
        """Create all necessary directories."""
        directories = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.mlflow_dir,
            self.base_dir / "logs",
            self.base_dir / "reports",
            self.base_dir / "artifacts"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {directory}")

# Example usage
if __name__ == "__main__":
    # Test logger
    logger = setup_logger("test_logger")
    logger.info("Logger setup complete")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Test paths
    paths = ProjectPaths()
    print(f"\\nProject Paths:")
    print(f"Raw data: {paths.raw_data_dir}")
    print(f"Processed data: {paths.processed_data_dir}")
    print(f"Models: {paths.models_dir}")

    # Create directories
    paths.create_directories()
