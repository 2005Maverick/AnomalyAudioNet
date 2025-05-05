import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """
    Load configuration from JSON file
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def save_config(config: dict, config_path: str):
    """
    Save configuration to JSON file
    
    Args:
        config (dict): Configuration dictionary
        config_path (str): Path to save config file
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving config: {str(e)}")
        raise

def create_directory(path: str):
    """
    Create directory if it doesn't exist
    
    Args:
        path (str): Directory path
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory: {str(e)}")
        raise

def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed (int): Random seed
    """
    np.random.seed(seed)
    # Add other random seed settings if needed
    # Example: tf.random.set_seed(seed) 