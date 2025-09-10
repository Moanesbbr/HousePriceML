"""
Configuration file for House Price Prediction project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# Data configuration
DATA_CONFIG = {
    'dataset_name': 'house_data_more_logic.xlsx',
    'num_samples': 1000,  # Increased for better training
    'random_seed': 42,
    'test_size': 0.2,
    'cv_folds': 10  # More folds for better validation
}

# Model configuration
MODEL_CONFIG = {
    'random_seed': 42,
    'tune_hyperparameters': True,
    'save_models': True,
    'models_to_train': [
        'linear_regression',
        'ridge',
        'lasso',
        'elastic_net',
        'random_forest',
        'gradient_boosting'
    ]
}

# Hyperparameter grids
PARAM_GRIDS = {
    'ridge': {
        'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    },
    'lasso': {
        'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    },
    'elastic_net': {
        'alpha': [0.01, 0.1, 0.5, 1.0, 5.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    'random_forest': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 15, 20, 25],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2, 3]
    },
    'gradient_boosting': {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.8, 0.9, 1.0]
    }
}

# Feature configuration
FEATURE_CONFIG = {
    'numerical_features': [
        'num_rooms', 'surface', 'num_bathrooms', 'age_of_house',
        'proximity_to_city_center_km', 'garden_size_sqm'
    ],
    'categorical_features': ['property_type'],
    'binary_features': ['has_kitchen', 'renovated', 'has_parking'],
    'ordinal_features': ['neighborhood_quality'],
    'include_engineered_features': True
}

# Visualization configuration
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'save_plots': True,
    'plot_format': 'png',
    'color_palette': 'husl'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': PROJECT_ROOT / 'logs' / 'house_price_ml.log'
}

# Ensure logs directory exists
LOGGING_CONFIG['log_file'].parent.mkdir(exist_ok=True)
