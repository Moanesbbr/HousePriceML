"""
Model Training Module for House Price Prediction

This module contains functions for training, evaluating, and saving
machine learning models for house price prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import json
from typing import Dict, List, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HousePriceModelTrainer:
    """
    A class to handle model training and evaluation for house price prediction.
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Define available models
        self.available_models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=42),
            'lasso': Lasso(random_state=42),
            'elastic_net': ElasticNet(random_state=42),
            'random_forest': RandomForestRegressor(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
        }
        
        # Define hyperparameter grids for tuning (optimized for better precision)
        self.param_grids = {
            'ridge': {'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]},
            'lasso': {'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]},
            'elastic_net': {'alpha': [0.01, 0.1, 0.5, 1.0, 5.0], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]},
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
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: pd.Series,
                          tune_hyperparameters: bool = False) -> Any:
        """
        Train a single model.
        
        Args:
            model_name (str): Name of the model to train
            X_train (np.ndarray): Training features
            y_train (pd.Series): Training target
            tune_hyperparameters (bool): Whether to tune hyperparameters
            
        Returns:
            Trained model
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(self.available_models.keys())}")
        
        logger.info(f"Training {model_name}...")
        
        if tune_hyperparameters and model_name in self.param_grids:
            # Hyperparameter tuning
            base_model = self.available_models[model_name]
            grid_search = GridSearchCV(
                base_model, 
                self.param_grids[model_name],
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        else:
            # Standard training
            model = self.available_models[model_name]
            model.fit(X_train, y_train)
        
        self.models[model_name] = model
        logger.info(f"✅ {model_name} training completed!")
        
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: pd.Series,
                        tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        Train all available models.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (pd.Series): Training target
            tune_hyperparameters (bool): Whether to tune hyperparameters
            
        Returns:
            Dict[str, Any]: Dictionary of trained models
        """
        logger.info("Training all models...")
        
        for model_name in self.available_models.keys():
            try:
                self.train_single_model(model_name, X_train, y_train, tune_hyperparameters)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
        
        logger.info(f"Completed training {len(self.models)} models")
        return self.models
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: pd.Series,
                      model_name: str) -> Dict[str, float]:
        """
        Evaluate a single model.
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        self.results[model_name] = metrics
        
        logger.info(f"Evaluation metrics for {model_name}:")
        logger.info(f"  RMSE: ${rmse:,.2f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAE: ${mae:,.2f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate all trained models.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (pd.Series): Test target
            
        Returns:
            pd.DataFrame: Results summary
        """
        logger.info("Evaluating all models...")
        
        for model_name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('rmse')
        
        # Identify best model
        self.best_model_name = results_df.index[0]
        self.best_model = self.models[self.best_model_name]
        
        logger.info(f"🏆 Best model: {self.best_model_name}")
        logger.info(f"Best RMSE: ${results_df.loc[self.best_model_name, 'rmse']:,.2f}")
        
        return results_df
    
    def cross_validate_models(self, X_train: np.ndarray, y_train: pd.Series,
                             cv_folds: int = 5) -> pd.DataFrame:
        """
        Perform cross-validation on all models.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (pd.Series): Training target
            cv_folds (int): Number of CV folds
            
        Returns:
            pd.DataFrame: Cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Convert to RMSE
            cv_rmse = np.sqrt(-cv_scores)
            
            cv_results[model_name] = {
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std(),
                'cv_rmse_scores': cv_rmse
            }
            
            logger.info(f"{model_name} CV RMSE: ${cv_rmse.mean():,.2f} ± ${cv_rmse.std():,.2f}")
        
        # Create results DataFrame
        cv_df = pd.DataFrame({
            name: [results['cv_rmse_mean'], results['cv_rmse_std']]
            for name, results in cv_results.items()
        }, index=['CV_RMSE_Mean', 'CV_RMSE_Std']).T
        
        cv_df = cv_df.sort_values('CV_RMSE_Mean')
        
        return cv_df
    
    def plot_predictions(self, X_test: np.ndarray, y_test: pd.Series,
                        model_name: str = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot predictions vs actual values.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (pd.Series): Test target
            model_name (str): Specific model to plot (if None, plots best model)
            figsize (Tuple[int, int]): Figure size
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Predictions vs Actual
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Prices')
        axes[0, 0].set_ylabel('Predicted Prices')
        axes[0, 0].set_title(f'Actual vs Predicted Prices - {model_name}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals vs Predicted
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Prices')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot of residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot comparison of all models.
        
        Args:
            figsize (Tuple[int, int]): Figure size
        """
        if not self.results:
            logger.warning("No results available. Train and evaluate models first.")
            return
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('rmse')
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # RMSE comparison
        axes[0, 0].bar(results_df.index, results_df['rmse'], color='skyblue', alpha=0.7)
        axes[0, 0].set_title('RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[0, 1].bar(results_df.index, results_df['r2'], color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('R² Comparison')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[1, 0].bar(results_df.index, results_df['mae'], color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('MAE Comparison')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        axes[1, 1].bar(results_df.index, results_df['mape'], color='gold', alpha=0.7)
        axes[1, 1].set_title('MAPE Comparison')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, feature_names: List[str] = None,
                              model_name: str = None, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Args:
            feature_names (List[str]): Names of features
            model_name (str): Specific model (if None, uses best model)
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature_importances_ attribute")
            return None
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def plot_feature_importance(self, feature_names: List[str] = None,
                               model_name: str = None, top_n: int = 15,
                               figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature importance.
        
        Args:
            feature_names (List[str]): Names of features
            model_name (str): Specific model (if None, uses best model)
            top_n (int): Number of top features to plot
            figsize (Tuple[int, int]): Figure size
        """
        importance_df = self.get_feature_importance(feature_names, model_name, top_n)
        
        if importance_df is None:
            return
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(importance_df)), importance_df['importance'], alpha=0.7)
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance')
        plt.title(f'Feature Importance - {model_name or self.best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_name: str = None, filepath: str = None):
        """
        Save a trained model.
        
        Args:
            model_name (str): Name of model to save (if None, saves best model)
            filepath (str): Path to save the model
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"../models/{model_name}_{timestamp}.joblib"
        
        joblib.dump(model, filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.results.get(model_name, {}),
            'model_type': type(model).__name__
        }
        
        metadata_path = filepath.replace('.joblib', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to {metadata_path}")
    
    def load_model(self, filepath: str):
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model
        """
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model


def main():
    """Main function to demonstrate model training."""
    # This would typically be called with preprocessed data
    logger.info("Model trainer initialized. Use with preprocessed data from data_preprocessor.py")


if __name__ == "__main__":
    main()
