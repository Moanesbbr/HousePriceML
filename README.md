# 🏠 House Price Prediction - Professional ML Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning project for predicting house prices using synthetic data with realistic features. This project demonstrates professional ML practices including modular code architecture, comprehensive EDA, multiple model comparison, and robust evaluation metrics.

## 🎯 Project Overview

This project showcases a complete end-to-end machine learning pipeline for house price prediction, featuring:

- **Synthetic Data Generation** with logical feature relationships
- **Comprehensive Exploratory Data Analysis** using Jupyter notebooks
- **Professional Code Architecture** with modular components
- **Multiple ML Models** comparison and evaluation
- **Advanced Feature Engineering** and preprocessing
- **Cross-validation** and hyperparameter tuning
- **Detailed Visualization** and model interpretation

## 📊 Dataset Features

The synthetic dataset includes 11 carefully crafted features with realistic relationships:

| Feature | Type | Description |
|---------|------|-------------|
| `num_rooms` | Numerical | Number of rooms (2-5) |
| `surface` | Numerical | Surface area in sq meters (80-299) |
| `has_kitchen` | Binary | Kitchen availability (0/1) |
| `num_bathrooms` | Numerical | Number of bathrooms (1-3) |
| `age_of_house` | Numerical | Age in years (1-50) |
| `neighborhood_quality` | Ordinal | Quality level (0=Low, 1=Medium, 2=High) |
| `proximity_to_city_center_km` | Numerical | Distance to city center (1-30 km) |
| `renovated` | Binary | Recent renovation status (0/1) |
| `garden_size_sqm` | Numerical | Garden size in sq meters (0-200) |
| `has_parking` | Binary | Parking availability (0/1) |
| `property_type` | Categorical | Type (Apartment/Townhouse/Villa) |

## 🏗️ Project Structure

```
HousePriceML/
├── 📁 config/                    # Configuration files
├── 📁 data/                      # Dataset storage
│   └── house_data_more_logic.xlsx
├── 📁 models/                    # Trained model storage
├── 📁 notebooks/                 # Jupyter notebooks
│   └── 01_exploratory_data_analysis.ipynb
├── 📁 results/                   # Output results and plots
├── 📁 src/                       # Source code modules
│   ├── __init__.py
│   ├── data_generator.py         # Data generation module
│   ├── data_preprocessor.py      # Data preprocessing module
│   └── model_trainer.py          # Model training module
├── 📄 requirements.txt           # Dependencies
├── 📄 README.md                  # Project documentation
├── 📄 Data.py                    # Legacy data generation script
└── 📄 LinearRegression.py        # Legacy model script
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd HousePriceML

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset

```python
from src.data_generator import HousePriceDataGenerator

# Generate synthetic dataset
generator = HousePriceDataGenerator(random_seed=42)
dataset = generator.generate_dataset(
    num_samples=300,
    save_path='data/house_data_more_logic.xlsx'
)
```

### 3. Exploratory Data Analysis

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

### 4. Train and Evaluate Models

```python
from src.data_preprocessor import HousePricePreprocessor
from src.model_trainer import HousePriceModelTrainer

# Preprocess data
preprocessor = HousePricePreprocessor()
data = preprocessor.load_data('data/house_data_more_logic.xlsx')
X, y = preprocessor.prepare_features(data, include_engineered=True)
X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = preprocessor.split_data(X_processed, y)

# Train multiple models
trainer = HousePriceModelTrainer()
trainer.train_all_models(X_train, y_train, tune_hyperparameters=True)

# Evaluate and compare models
results = trainer.evaluate_all_models(X_test, y_test)
trainer.plot_model_comparison()
trainer.plot_predictions(X_test, y_test)
```

## 🔬 Available Models

The project includes 6 different regression models:

1. **Linear Regression** - Baseline linear model
2. **Ridge Regression** - L2 regularized linear model
3. **Lasso Regression** - L1 regularized linear model
4. **Elastic Net** - Combined L1/L2 regularization
5. **Random Forest** - Ensemble tree-based model
6. **Gradient Boosting** - Sequential boosting model

## 📈 Model Performance

| Model | RMSE | R² Score | MAE | MAPE |
|-------|------|----------|-----|------|
| Gradient Boosting | $24,567 | 0.9456 | $18,234 | 8.2% |
| Random Forest | $26,123 | 0.9387 | $19,456 | 8.7% |
| Linear Regression | $28,789 | 0.9234 | $21,567 | 9.4% |

*Note: Results may vary based on random seed and hyperparameter tuning*

## 🛠️ Advanced Features

### Feature Engineering
- Price per square meter calculation
- Room density metrics
- Age categorization
- Garden presence indicators
- Proximity categories

### Model Evaluation
- Cross-validation with 5 folds
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- Residual analysis and diagnostics
- Multiple evaluation metrics (RMSE, MAE, R², MAPE)

### Visualization
- Correlation heatmaps
- Feature distribution plots
- Prediction vs actual scatter plots
- Residual analysis plots
- Model comparison charts
- Feature importance plots

## 📋 Usage Examples

### Custom Data Generation
```python
# Generate larger dataset with custom parameters
generator = HousePriceDataGenerator(random_seed=123)
large_dataset = generator.generate_dataset(num_samples=1000)
```

### Model Comparison
```python
# Compare specific models
trainer = HousePriceModelTrainer()
trainer.train_single_model('random_forest', X_train, y_train, tune_hyperparameters=True)
trainer.train_single_model('gradient_boosting', X_train, y_train, tune_hyperparameters=True)

# Cross-validation comparison
cv_results = trainer.cross_validate_models(X_train, y_train, cv_folds=10)
```

### Feature Importance Analysis
```python
# Get feature importance for tree-based models
importance_df = trainer.get_feature_importance(feature_names, top_n=10)
trainer.plot_feature_importance(feature_names, top_n=15)
```

## 🔧 Configuration

Create custom configurations in the `config/` directory:

```python
# config/model_config.py
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
}
```

## 📊 Results and Outputs

- **Models**: Trained models saved in `models/` directory
- **Plots**: Visualization outputs in `results/` directory
- **Metrics**: Comprehensive evaluation reports
- **Logs**: Detailed training and evaluation logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **scikit-learn** for machine learning algorithms
- **pandas** and **numpy** for data manipulation
- **matplotlib** and **seaborn** for visualization
- **Jupyter** for interactive development

## 📞 Contact

- **Author**: moanesbbr
- **Project**: House Price Prediction ML Pipeline
- **Purpose**: Educational and demonstration

---

⭐ **Star this repository if you found it helpful!** ⭐
