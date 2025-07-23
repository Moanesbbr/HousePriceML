import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# --- Generate Synthetic Data with More Logic Columns ---
num_samples = 300

# Basic Features (from previous example)
num_rooms = np.random.randint(2, 6, num_samples)  # Number of rooms (2-5)
surface = np.random.randint(80, 300, num_samples) # Surface area in sq meters (80-299)
has_kitchen = np.random.choice([0, 1], num_samples, p=[0.1, 0.9]) # 1 if has kitchen, 0 otherwise
num_bathrooms = np.random.randint(1, 4, num_samples) # Number of bathrooms (1-3)
age_of_house = np.random.randint(1, 50, num_samples) # Age of the house in years

# --- New Logic-based Columns ---

# 1. Neighborhood Quality (0: Low, 1: Medium, 2: High)
neighborhood_quality = np.random.choice([0, 1, 2], num_samples, p=[0.2, 0.5, 0.3])
# This will have a strong positive correlation with price.

# 2. Proximity to City Center (in km, lower is better)
# Assume 'High' quality neighborhoods are closer, 'Low' are further.
proximity_to_city_center = np.zeros(num_samples)
for i in range(num_samples):
    if neighborhood_quality[i] == 2: # High quality
        proximity_to_city_center[i] = np.random.uniform(1, 5) # 1-5 km
    elif neighborhood_quality[i] == 1: # Medium quality
        proximity_to_city_center[i] = np.random.uniform(4, 15) # 4-15 km
    else: # Low quality
        proximity_to_city_center[i] = np.random.uniform(10, 30) # 10-30 km
# This will have a negative correlation with price.

# 3. Renovation Status (0: No recent reno, 1: Recently renovated)
renovated = np.random.choice([0, 1], num_samples, p=[0.7, 0.3])
# Renovated houses get a price premium.

# 4. Garden/Yard Size (in sq meters, 0 if no garden)
garden_size = np.zeros(num_samples)
# Let's say 40% of houses have a garden
has_garden_indices = np.random.choice(num_samples, int(num_samples * 0.4), replace=False)
garden_size[has_garden_indices] = np.random.randint(10, 200, len(has_garden_indices))
# Positive correlation with price.

# 5. Parking Availability (0: No dedicated, 1: Has dedicated parking)
has_parking = np.random.choice([0, 1], num_samples, p=[0.3, 0.7])
# Houses with parking get a premium.

# 6. Property Type (Using one-hot encoding implicitly for price calculation)
# 0: Apartment, 1: Townhouse, 2: Villa
property_type_codes = np.random.choice([0, 1, 2], num_samples, p=[0.4, 0.3, 0.3])
property_type_map = {0: 'Apartment', 1: 'Townhouse', 2: 'Villa'}
property_type = np.array([property_type_map[code] for code in property_type_codes])

# --- Target variable (Price) - adjusted with new features ---
# Price = (coeff_basic * basic_features) + (coeff_new * new_features) + noise

# Base coefficients for the original features
coeff_num_rooms = 50000
coeff_surface = 1500
coeff_has_kitchen = 20000
coeff_num_bathrooms = 10000
coeff_age_of_house = -500 # Older houses are generally cheaper

# Coefficients for new logic columns
coeff_neighborhood_quality = [0, 50000, 150000] # Price premium for each quality level
coeff_proximity_to_city_center = -3000 # Each km further decreases price
coeff_renovated = 30000 # Premium for renovated houses
coeff_garden_size = 500 # Price per sq meter of garden
coeff_has_parking = 15000 # Premium for parking

# Property type base price adjustment (apartments are base, townhouses more, villas most)
property_type_price_adjust = np.zeros(num_samples)
for i in range(num_samples):
    if property_type_codes[i] == 1: # Townhouse
        property_type_price_adjust[i] = 50000
    elif property_type_codes[i] == 2: # Villa
        property_type_price_adjust[i] = 150000

# Calculate price
price = (coeff_num_rooms * num_rooms +
         coeff_surface * surface +
         coeff_has_kitchen * has_kitchen +
         coeff_num_bathrooms * num_bathrooms +
         coeff_age_of_house * age_of_house +
         np.array([coeff_neighborhood_quality[q] for q in neighborhood_quality]) +
         coeff_proximity_to_city_center * proximity_to_city_center +
         coeff_renovated * renovated +
         coeff_garden_size * garden_size +
         coeff_has_parking * has_parking +
         property_type_price_adjust +
         np.random.normal(0, 25000, num_samples)) # Increased noise for more variance

# Ensure prices are not negative and have a reasonable floor
price[price < 80000] = 80000 + np.random.normal(0, 10000, np.sum(price < 80000))

# Create a Pandas DataFrame
data = pd.DataFrame({
    'num_rooms': num_rooms,
    'surface': surface,
    'has_kitchen': has_kitchen,
    'num_bathrooms': num_bathrooms,
    'age_of_house': age_of_house,
    'neighborhood_quality': neighborhood_quality, # Ordinal: 0, 1, 2
    'proximity_to_city_center_km': proximity_to_city_center,
    'renovated': renovated, # Binary: 0 or 1
    'garden_size_sqm': garden_size,
    'has_parking': has_parking, # Binary: 0 or 1
    'property_type': property_type, # Categorical: Apartment, Townhouse, Villa
    'price': price
})

# Display first few rows and basic info
print("--- Generated Data Head ---")
print(data.head())
print("\n--- Data Info ---")
data.info()
print("\n--- Data Description ---")
print(data.describe())

# --- Save the DataFrame to an Excel file ---
excel_file_name = 'house_data_more_logic.xlsx'
data.to_excel(excel_file_name, index=False)

print(f"\nData successfully saved to '{excel_file_name}'")
print("You can now open this file in Microsoft Excel or any spreadsheet software.")
print("The file will be saved in the same directory where you run this Python script.")