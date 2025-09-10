"""
Data Generation Module for House Price Prediction

This module contains functions to generate synthetic house price data
with realistic features and logical relationships.
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HousePriceDataGenerator:
    """
    A class to generate synthetic house price data with realistic features.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            random_seed (int): Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Define coefficients for price calculation
        self.price_coefficients = {
            'num_rooms': 50000,
            'surface': 1500,
            'has_kitchen': 20000,
            'num_bathrooms': 10000,
            'age_of_house': -500,
            'neighborhood_quality': [0, 50000, 150000],
            'proximity_to_city_center': -3000,
            'renovated': 30000,
            'garden_size': 500,
            'has_parking': 15000,
            'property_type': {'Apartment': 0, 'Townhouse': 50000, 'Villa': 150000}
        }
    
    def generate_basic_features(self, num_samples: int) -> Dict[str, np.ndarray]:
        """
        Generate basic house features.
        
        Args:
            num_samples (int): Number of samples to generate
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing basic features
        """
        return {
            'num_rooms': np.random.randint(2, 6, num_samples),
            'surface': np.random.randint(80, 300, num_samples),
            'has_kitchen': np.random.choice([0, 1], num_samples, p=[0.1, 0.9]),
            'num_bathrooms': np.random.randint(1, 4, num_samples),
            'age_of_house': np.random.randint(1, 50, num_samples)
        }
    
    def generate_neighborhood_features(self, num_samples: int) -> Dict[str, np.ndarray]:
        """
        Generate neighborhood-related features with logical relationships.
        
        Args:
            num_samples (int): Number of samples to generate
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing neighborhood features
        """
        # Neighborhood quality (0: Low, 1: Medium, 2: High)
        neighborhood_quality = np.random.choice([0, 1, 2], num_samples, p=[0.2, 0.5, 0.3])
        
        # Proximity to city center based on neighborhood quality
        proximity_to_city_center = np.zeros(num_samples)
        for i in range(num_samples):
            if neighborhood_quality[i] == 2:  # High quality
                proximity_to_city_center[i] = np.random.uniform(1, 5)
            elif neighborhood_quality[i] == 1:  # Medium quality
                proximity_to_city_center[i] = np.random.uniform(4, 15)
            else:  # Low quality
                proximity_to_city_center[i] = np.random.uniform(10, 30)
        
        return {
            'neighborhood_quality': neighborhood_quality,
            'proximity_to_city_center_km': proximity_to_city_center
        }
    
    def generate_property_features(self, num_samples: int) -> Dict[str, np.ndarray]:
        """
        Generate property-specific features.
        
        Args:
            num_samples (int): Number of samples to generate
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing property features
        """
        # Renovation status
        renovated = np.random.choice([0, 1], num_samples, p=[0.7, 0.3])
        
        # Garden size (40% of houses have gardens)
        garden_size = np.zeros(num_samples)
        has_garden_indices = np.random.choice(num_samples, int(num_samples * 0.4), replace=False)
        garden_size[has_garden_indices] = np.random.randint(10, 200, len(has_garden_indices))
        
        # Parking availability
        has_parking = np.random.choice([0, 1], num_samples, p=[0.3, 0.7])
        
        # Property type
        property_type_codes = np.random.choice([0, 1, 2], num_samples, p=[0.4, 0.3, 0.3])
        property_type_map = {0: 'Apartment', 1: 'Townhouse', 2: 'Villa'}
        property_type = np.array([property_type_map[code] for code in property_type_codes])
        
        return {
            'renovated': renovated,
            'garden_size_sqm': garden_size,
            'has_parking': has_parking,
            'property_type': property_type,
            'property_type_codes': property_type_codes
        }
    
    def calculate_price(self, features: Dict[str, np.ndarray], num_samples: int) -> np.ndarray:
        """
        Calculate house prices based on features and coefficients.
        
        Args:
            features (Dict[str, np.ndarray]): Dictionary containing all features
            num_samples (int): Number of samples
            
        Returns:
            np.ndarray: Calculated prices
        """
        # Base price calculation
        price = (
            self.price_coefficients['num_rooms'] * features['num_rooms'] +
            self.price_coefficients['surface'] * features['surface'] +
            self.price_coefficients['has_kitchen'] * features['has_kitchen'] +
            self.price_coefficients['num_bathrooms'] * features['num_bathrooms'] +
            self.price_coefficients['age_of_house'] * features['age_of_house'] +
            np.array([self.price_coefficients['neighborhood_quality'][q] 
                     for q in features['neighborhood_quality']]) +
            self.price_coefficients['proximity_to_city_center'] * features['proximity_to_city_center_km'] +
            self.price_coefficients['renovated'] * features['renovated'] +
            self.price_coefficients['garden_size'] * features['garden_size_sqm'] +
            self.price_coefficients['has_parking'] * features['has_parking']
        )
        
        # Add property type adjustment
        property_type_adjustment = np.zeros(num_samples)
        for i, prop_type in enumerate(features['property_type']):
            property_type_adjustment[i] = self.price_coefficients['property_type'][prop_type]
        
        price += property_type_adjustment
        
        # Add minimal noise for more precise predictions
        price += np.random.normal(0, 5000, num_samples)
        
        # Ensure minimum price with less noise
        price[price < 80000] = 80000 + np.random.normal(0, 2000, np.sum(price < 80000))
        
        return price
    
    def generate_dataset(self, num_samples: int = 300, save_path: str = None) -> pd.DataFrame:
        """
        Generate complete synthetic house price dataset.
        
        Args:
            num_samples (int): Number of samples to generate
            save_path (str): Path to save the dataset (optional)
            
        Returns:
            pd.DataFrame: Generated dataset
        """
        logger.info(f"Generating synthetic dataset with {num_samples} samples...")
        
        # Generate all features
        basic_features = self.generate_basic_features(num_samples)
        neighborhood_features = self.generate_neighborhood_features(num_samples)
        property_features = self.generate_property_features(num_samples)
        
        # Combine all features
        all_features = {**basic_features, **neighborhood_features, **property_features}
        
        # Calculate prices
        prices = self.calculate_price(all_features, num_samples)
        all_features['price'] = prices
        
        # Remove helper columns
        all_features.pop('property_type_codes', None)
        
        # Create DataFrame
        data = pd.DataFrame(all_features)
        
        logger.info("Dataset generation completed successfully!")
        logger.info(f"Dataset shape: {data.shape}")
        logger.info(f"Price range: ${data['price'].min():,.0f} - ${data['price'].max():,.0f}")
        
        # Save if path provided
        if save_path:
            data.to_excel(save_path, index=False)
            logger.info(f"Dataset saved to: {save_path}")
        
        return data


def main():
    """Main function to generate and save dataset."""
    generator = HousePriceDataGenerator(random_seed=42)
    
    # Generate dataset
    dataset = generator.generate_dataset(
        num_samples=300,
        save_path='../data/house_data_more_logic.xlsx'
    )
    
    # Display basic information
    print("\n--- Generated Dataset Overview ---")
    print(dataset.head())
    print("\n--- Dataset Info ---")
    dataset.info()
    print("\n--- Dataset Description ---")
    print(dataset.describe())


if __name__ == "__main__":
    main()
