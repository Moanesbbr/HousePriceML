"""
Data Preprocessing Module for House Price Prediction

This module contains functions for data cleaning, feature engineering,
and preprocessing for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HousePricePreprocessor:
    """
    A class to handle data preprocessing for house price prediction.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.preprocessor = None
        self.feature_names = None
        self.numerical_features = [
            'num_rooms', 'surface', 'num_bathrooms', 'age_of_house',
            'proximity_to_city_center_km', 'garden_size_sqm'
        ]
        self.categorical_features = ['property_type']
        self.binary_features = ['has_kitchen', 'renovated', 'has_parking']
        self.ordinal_features = ['neighborhood_quality']
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Args:
            file_path (str): Path to the Excel file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            data = pd.read_excel(file_path)
            logger.info(f"Data loaded successfully from {file_path}")
            logger.info(f"Dataset shape: {data.shape}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform data quality checks.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            Dict[str, Any]: Data quality report
        """
        quality_report = {
            'shape': data.shape,
            'missing_values': data.isnull().sum().to_dict(),
            'duplicates': data.duplicated().sum(),
            'data_types': data.dtypes.to_dict()
        }
        
        logger.info("Data Quality Report:")
        logger.info(f"  Shape: {quality_report['shape']}")
        logger.info(f"  Missing values: {sum(quality_report['missing_values'].values())}")
        logger.info(f"  Duplicates: {quality_report['duplicates']}")
        
        return quality_report
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with handled missing values
        """
        data_clean = data.copy()
        
        # Handle numerical columns
        for col in self.numerical_features:
            if col in data_clean.columns and data_clean[col].isnull().any():
                mean_value = data_clean[col].mean()
                data_clean[col].fillna(mean_value, inplace=True)
                logger.info(f"Filled missing values in {col} with mean: {mean_value:.2f}")
        
        # Handle categorical columns
        categorical_cols = self.categorical_features + self.binary_features + self.ordinal_features
        for col in categorical_cols:
            if col in data_clean.columns and data_clean[col].isnull().any():
                mode_value = data_clean[col].mode()[0]
                data_clean[col].fillna(mode_value, inplace=True)
                logger.info(f"Filled missing values in {col} with mode: {mode_value}")
        
        return data_clean
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for better model performance.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with additional features
        """
        data_enhanced = data.copy()
        
        # Price per square meter (only if price column exists)
        if 'price' in data_enhanced.columns:
            data_enhanced['price_per_sqm'] = data_enhanced['price'] / data_enhanced['surface']
        
        # Room density and efficiency metrics
        data_enhanced['room_density'] = data_enhanced['num_rooms'] / data_enhanced['surface']
        data_enhanced['bathroom_ratio'] = data_enhanced['num_bathrooms'] / data_enhanced['num_rooms']
        data_enhanced['total_rooms'] = data_enhanced['num_rooms'] + data_enhanced['num_bathrooms']
        
        # Garden and outdoor features
        data_enhanced['has_garden'] = (data_enhanced['garden_size_sqm'] > 0).astype(int)
        data_enhanced['garden_to_surface_ratio'] = data_enhanced['garden_size_sqm'] / data_enhanced['surface']
        
        # Location and accessibility score
        data_enhanced['location_score'] = (
            data_enhanced['neighborhood_quality'] * 2 - 
            np.log1p(data_enhanced['proximity_to_city_center_km']) + 
            data_enhanced['has_parking'] * 0.5
        )
        
        # Property value indicators
        data_enhanced['luxury_score'] = (
            (data_enhanced['surface'] > 200).astype(int) +
            (data_enhanced['num_rooms'] >= 4).astype(int) +
            (data_enhanced['num_bathrooms'] >= 2).astype(int) +
            data_enhanced['renovated'] +
            data_enhanced['has_garden']
        )
        
        # Age-related features
        data_enhanced['is_new'] = (data_enhanced['age_of_house'] <= 5).astype(int)
        data_enhanced['is_old'] = (data_enhanced['age_of_house'] >= 30).astype(int)
        data_enhanced['age_squared'] = data_enhanced['age_of_house'] ** 2
        
        # Surface efficiency
        data_enhanced['surface_efficiency'] = data_enhanced['num_rooms'] * data_enhanced['surface'] / 100
        
        # Interaction features
        data_enhanced['renovated_x_age'] = data_enhanced['renovated'] * data_enhanced['age_of_house']
        data_enhanced['quality_x_surface'] = data_enhanced['neighborhood_quality'] * data_enhanced['surface']
        
        # Categorical features (keeping existing ones)
        data_enhanced['age_category'] = pd.cut(
            data_enhanced['age_of_house'], 
            bins=[0, 10, 25, 50], 
            labels=['New', 'Medium', 'Old'],
            include_lowest=True
        )
        
        data_enhanced['surface_category'] = pd.cut(
            data_enhanced['surface'],
            bins=[0, 100, 150, 200, 300],
            labels=['Small', 'Medium', 'Large', 'XLarge'],
            include_lowest=True
        )
        
        data_enhanced['proximity_category'] = pd.cut(
            data_enhanced['proximity_to_city_center_km'],
            bins=[0, 5, 15, 30],
            labels=['Close', 'Medium', 'Far'],
            include_lowest=True
        )
        
        logger.info("Created enhanced features for better precision:")
        logger.info("  - Basic: room_density, bathroom_ratio, total_rooms")
        logger.info("  - Garden: has_garden, garden_to_surface_ratio")
        logger.info("  - Location: location_score, luxury_score")
        logger.info("  - Age: is_new, is_old, age_squared")
        logger.info("  - Interactions: renovated_x_age, quality_x_surface")
        
        return data_enhanced
    
    def prepare_features(self, data: pd.DataFrame, include_engineered: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable.
        
        Args:
            data (pd.DataFrame): Input dataset
            include_engineered (bool): Whether to include engineered features
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target variable
        """
        # Create engineered features if requested
        if include_engineered:
            data = self.create_features(data)
        
        # Separate features and target
        if 'price' in data.columns:
            X = data.drop('price', axis=1)
            y = data['price']
        else:
            X = data
            y = None
        
        # Update feature lists if engineered features are included
        if include_engineered:
            self.numerical_features.extend([
                'room_density', 'bathroom_ratio', 'total_rooms', 'price_per_sqm',
                'garden_to_surface_ratio', 'location_score', 'luxury_score',
                'age_squared', 'surface_efficiency', 'renovated_x_age', 'quality_x_surface'
            ])
            self.binary_features.extend(['has_garden', 'is_new', 'is_old'])
            self.categorical_features.extend([
                'age_category', 'surface_category', 'proximity_category'
            ])
        
        return X, y
    
    def create_preprocessor(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline.
        
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        # Define transformations
        transformers = [
            ('num', StandardScaler(), self.numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), self.categorical_features)
        ]
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'  # Keep binary and ordinal features as-is
        )
        
        return preprocessor
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit preprocessor and transform features.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Transformed features
        """
        self.preprocessor = self.create_preprocessor()
        X_transformed = self.preprocessor.fit_transform(X)
        
        logger.info(f"Features transformed. Shape: {X_transformed.shape}")
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted preprocessor.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Transformed features
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        X_transformed = self.preprocessor.transform(X)
        return X_transformed
    
    def split_data(self, X: np.ndarray, y: pd.Series, test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X (np.ndarray): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Training set: {X_train.shape[0]} samples")
        logger.info(f"  Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names after preprocessing.
        
        Returns:
            List[str]: Feature names
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted.")
        
        feature_names = []
        
        # Get feature names from each transformer
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend([f"num_{feat}" for feat in features])
            elif name == 'cat':
                if hasattr(transformer, 'get_feature_names_out'):
                    cat_features = transformer.get_feature_names_out(features)
                    feature_names.extend(cat_features)
                else:
                    feature_names.extend([f"cat_{feat}" for feat in features])
        
        # Add remainder features
        remainder_features = [col for col in self.binary_features + self.ordinal_features]
        feature_names.extend(remainder_features)
        
        return feature_names


def main():
    """Main function to demonstrate preprocessing."""
    preprocessor = HousePricePreprocessor()
    
    # Load data
    data = preprocessor.load_data('../data/house_data_more_logic.xlsx')
    
    # Check data quality
    quality_report = preprocessor.check_data_quality(data)
    
    # Handle missing values
    data_clean = preprocessor.handle_missing_values(data)
    
    # Prepare features
    X, y = preprocessor.prepare_features(data_clean, include_engineered=True)
    
    # Transform features
    X_transformed = preprocessor.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_transformed, y)
    
    print(f"\nPreprocessing completed successfully!")
    print(f"Final feature shape: {X_transformed.shape}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")


if __name__ == "__main__":
    main()
