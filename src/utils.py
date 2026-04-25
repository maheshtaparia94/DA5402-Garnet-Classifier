import logging
import os
from pathlib import Path
import yaml



def load_config(config_path: str = "params.yaml") -> dict:
    """
    Load pipeline configuration from params.yaml.

    Args:
        config_path: Path to params.yaml from project root.

    Returns:
        dict: Configuration parameters.
    
    Raises:
        FileNotFoundError: If params.yaml does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config



def get_logger(name: str) -> logging.Logger:
    """
    Get a logger that writes to console and logs/ folder.

    Args:
        name: Logger name — use __name__ in calling script.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logs/ directory if it does not exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Log format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file = logs_dir / f"{name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
