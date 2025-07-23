# HousePriceML

A Machine Learning project for predicting house prices using a synthetic dataset with rich, realistic features. This repository demonstrates a complete workflow: data generation, preprocessing, model training, evaluation, and visualization using Python and scikit-learn.

## Description

**HousePriceML** is designed to showcase a practical approach to regression modeling for real estate price prediction. It includes:

- Synthetic data generation with logical, realistic features (e.g., rooms, surface, neighborhood quality, proximity to city center, renovation status, garden, parking, property type).
- Data cleaning, feature engineering, and preprocessing (scaling, encoding).
- Linear regression model training and evaluation.
- Visualization of predictions, residuals, and model performance metrics.

## Features

- **Synthetic Data Generation**: Generates a dataset with 300 samples and saves it as `house_data_more_logic.xlsx`.
- **Data Preprocessing**: Handles missing values, scales numerical features, and encodes categorical variables.
- **Model Training**: Trains a linear regression model to predict house prices.
- **Evaluation & Visualization**: Provides metrics (MSE, RMSE, R²) and plots for model assessment.

## File Structure

- `Data.py`: Generates the synthetic dataset and saves it as an Excel file.
- `house_data_more_logic.xlsx`: The generated dataset used for training and testing.
- `LinearRegression.py`: Loads the dataset, preprocesses data, trains the model, evaluates, and visualizes results.

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd HousePriceML
   ```
2. **Install dependencies**
   Ensure you have Python 3.7+ and install required packages:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn tqdm openpyxl
   ```
3. **Generate the dataset**
   Run the data generation script:
   ```bash
   python Data.py
   ```
4. **Train and evaluate the model**
   Run the main regression script:
   ```bash
   python LinearRegression.py
   ```

## Usage

- Modify `Data.py` to adjust the logic or number of samples for the synthetic dataset.
- Use `LinearRegression.py` to experiment with different preprocessing or modeling techniques.
- Visualizations and metrics will be displayed in the console and as plots.

## Credits

- Developed by moanesbbr.
- Built with Python, scikit-learn, pandas, matplotlib, and seaborn.

## License

This project is for educational and demonstration purposes.
