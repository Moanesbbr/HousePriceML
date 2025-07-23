# A Simple Linear Regression Model for House Price Prediction (using external Excel data)

# This notebook demonstrates a basic linear regression model to predict house prices
# using data loaded from an Excel file, including data cleaning,
# scaling, model training, and evaluation.

# --- Step 1: Import necessary libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm


# --- Step 2: Load Data from Excel File ---
# Ensure 'house_data_more_logic.xlsx' is in the same directory as this script
excel_file_name = 'house_data_more_logic.xlsx'
try:
    data = pd.read_excel(excel_file_name)
    print(f"--- Data loaded successfully from '{excel_file_name}' ---")
except FileNotFoundError:
    print(f"Error: The file '{excel_file_name}' was not found.")
    print("Please make sure the Excel file is in the same directory as this script.")
    exit() # Exit if the file is not found

print("\n--- Data Head ---")
print(data.head())
print("\n--- Data Info ---")
data.info()
print("\n--- Data Description ---")
print(data.describe(include='all')) # include='all' to see categorical descriptions too

# --- Step 3: Data Cleaning Functionalities ---
print("\n--- Performing a check for missing values ---")
print(data.isnull().sum())

# For this synthetic dataset, we assume missing values are handled by mean imputation for numericals.
# We'll explicitly define numerical and categorical columns for robust cleaning.

numerical_cols = [
    'num_rooms', 'surface', 'num_bathrooms', 'age_of_house',
    'proximity_to_city_center_km', 'garden_size_sqm'
]
categorical_cols = [
    'has_kitchen', 'neighborhood_quality', 'renovated',
    'has_parking', 'property_type'
]

# Impute missing numerical values with the mean
for col in numerical_cols:
    if data[col].isnull().any():
        data[col].fillna(data[col].mean(), inplace=True)
        print(f"Missing values in numerical column '{col}' imputed with mean.")

# For categorical columns, if there were missing values, you might fill with mode or 'unknown'.
# For our synthetic data, there should be no missing categorical values. most frequency value
for col in categorical_cols:
    if data[col].isnull().any():
        data[col].fillna(data[col].mode()[0], inplace=True) # Fill with mode
        print(f"Missing values in categorical column '{col}' imputed with mode.")

print("\n--- Data after imputing missing values ---")
print(data.isnull().sum())

# --- Step 4: Separate Features (X) and Target (y) ---
X = data.drop('price', axis=1)
y = data['price']

# Define which columns are numerical and which are categorical for preprocessing
numerical_features = [
    'num_rooms', 'surface', 'num_bathrooms', 'age_of_house',
    'proximity_to_city_center_km', 'garden_size_sqm'
]
# 'has_kitchen', 'renovated', 'has_parking' are binary (0/1) so they can be treated as numerical or categorical.
# For linear regression, they work well as numerical.
# 'neighborhood_quality' is ordinal (0,1,2). We can treat it as numerical or one-hot encode.
# For simplicity, and since the values imply an order, we'll keep it numerical for now.
# 'property_type' is truly nominal categorical and MUST be one-hot encoded.
categorical_features = ['property_type'] # Only this one truly needs OneHotEncoding for Linear Regression

# --- Step 5: Feature Scaling and Encoding using ColumnTransformer ---
# This is a powerful tool to apply different transformations to different columns.

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features), # b decent de radient for beeter performance and fst to get to target minimum
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (like has_kitchen, renovated, has_parking, neighborhood_quality) as they are for now
)

# Apply preprocessing to X
X_processed = preprocessor.fit_transform(X)

# Get the names of the processed columns for better understanding (optional but good practice)
# This part can be tricky with ColumnTransformer if not careful, especially with OneHotEncoder.
# For simplicity, we'll just acknowledge X_processed is a NumPy array for now.
# If you need column names, you'd extend this part.

print("\n--- Shape of Processed Features (X_processed) ---")
print(X_processed.shape) # Number of features will increase due to OneHotEncoding

# --- Step 6: Split Data into Training and Test Sets ---
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")


# --- Step 7: Create and Train the Linear Regression Model with loader ---
print("\n--- Training Linear Regression Model ---")
print("Training in progress...")

# Simulate loader while fitting
with tqdm(total=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}%', desc="Model Training") as pbar:
    for i in range(5):  # Simulate progress
        time.sleep(0.3)  # short delay to show progress
        pbar.update(20)
    model = LinearRegression()
    model.fit(X_train, y_train)
    pbar.update(100 - pbar.n)

print("✅ Model Training Complete!")
print(f"Intercept: {model.intercept_:.2f}")
print()

# --- Step 8: Make Predictions ---
y_pred = model.predict(X_test)

residuals = y_test - y_pred

# --- Step 9: Plotting Predictions vs. Actuals ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal', edgecolor='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label="Perfect Prediction")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.grid(True)
plt.legend()
plt.show()

# Residuals distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='orange')
plt.title("Residuals Distribution (Actual - Predicted)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Residuals vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7, color='purple', edgecolor='k')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted Prices")
plt.grid(True)
plt.show()

# --- Step 10: Evaluate the Model (Metrics) ---
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 --- Model Evaluation Metrics ---")
print(f"🔷 Mean Squared Error (MSE):       {mse:,.2f}")
print(f"🔷 Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"🔷 R-squared (R²):                 {r2:.2f}")
print()
print(f"✅ The RMSE of ${rmse:,.2f} means, on average, predictions are off by ~${rmse:,.2f}.")
print(f"✅ The R² of {r2:.2f} suggests about {r2*100:.2f}% of the price variance is explained by the model.")
print("🎯 Model Evaluation Complete!")

