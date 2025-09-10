"""
Main execution script for House Price Prediction ML Pipeline.

This script demonstrates the complete workflow from data generation
to model training and evaluation.
"""

import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_generator import HousePriceDataGenerator
from src.data_preprocessor import HousePricePreprocessor
from src.model_trainer import HousePriceModelTrainer
from config.config import *

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    logger.info("🏠 Starting House Price Prediction ML Pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 1: Generate Dataset
        logger.info("📊 Step 1: Generating synthetic dataset...")
        generator = HousePriceDataGenerator(random_seed=DATA_CONFIG['random_seed'])
        
        dataset_path = DATA_DIR / DATA_CONFIG['dataset_name']
        dataset = generator.generate_dataset(
            num_samples=DATA_CONFIG['num_samples'],
            save_path=str(dataset_path)
        )
        logger.info(f"✅ Dataset generated and saved to {dataset_path}")
        
        # Step 2: Data Preprocessing
        logger.info("\n🔧 Step 2: Preprocessing data...")
        preprocessor = HousePricePreprocessor()
        
        # Load and clean data
        data = preprocessor.load_data(str(dataset_path))
        data_clean = preprocessor.handle_missing_values(data)
        
        # Prepare features
        X, y = preprocessor.prepare_features(
            data_clean, 
            include_engineered=FEATURE_CONFIG['include_engineered_features']
        )
        
        # Transform features
        X_processed = preprocessor.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            X_processed, y,
            test_size=DATA_CONFIG['test_size'],
            random_state=DATA_CONFIG['random_seed']
        )
        logger.info("✅ Data preprocessing completed")
        
        # Step 3: Model Training
        logger.info("\n🤖 Step 3: Training multiple models...")
        trainer = HousePriceModelTrainer()
        
        # Train all models
        models = trainer.train_all_models(
            X_train, y_train,
            tune_hyperparameters=MODEL_CONFIG['tune_hyperparameters']
        )
        logger.info(f"✅ Trained {len(models)} models successfully")
        
        # Step 4: Model Evaluation
        logger.info("\n📈 Step 4: Evaluating models...")
        results_df = trainer.evaluate_all_models(X_test, y_test)
        
        # Cross-validation
        cv_results = trainer.cross_validate_models(
            X_train, y_train,
            cv_folds=DATA_CONFIG['cv_folds']
        )
        
        # Step 5: Results Visualization
        logger.info("\n📊 Step 5: Creating visualizations...")
        
        # Model comparison plots
        trainer.plot_model_comparison(figsize=VIZ_CONFIG['figure_size'])
        
        # Prediction plots for best model
        trainer.plot_predictions(
            X_test, y_test,
            figsize=VIZ_CONFIG['figure_size']
        )
        
        # Feature importance (if available)
        try:
            feature_names = preprocessor.get_feature_names()
            trainer.plot_feature_importance(
                feature_names=feature_names,
                figsize=VIZ_CONFIG['figure_size']
            )
        except Exception as e:
            logger.warning(f"Could not plot feature importance: {e}")
        
        # Step 6: Save Results
        logger.info("\n💾 Step 6: Saving results...")
        
        # Save best model
        if MODEL_CONFIG['save_models']:
            model_path = MODELS_DIR / f"best_model_{trainer.best_model_name}.joblib"
            trainer.save_model(filepath=str(model_path))
        
        # Save results
        results_path = RESULTS_DIR / "model_comparison_results.csv"
        results_df.to_csv(results_path)
        
        cv_results_path = RESULTS_DIR / "cross_validation_results.csv"
        cv_results.to_csv(cv_results_path)
        
        logger.info(f"✅ Results saved to {RESULTS_DIR}")
        
        # Step 7: Summary Report
        logger.info("\n📋 FINAL RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"🏆 Best Model: {trainer.best_model_name}")
        logger.info(f"📊 Dataset Size: {len(dataset)} samples")
        logger.info(f"🔧 Features: {X_processed.shape[1]} (after preprocessing)")
        logger.info(f"📈 Models Trained: {len(models)}")
        
        best_metrics = trainer.results[trainer.best_model_name]
        logger.info(f"\n🎯 Best Model Performance:")
        logger.info(f"   RMSE: ${best_metrics['rmse']:,.2f}")
        logger.info(f"   R² Score: {best_metrics['r2']:.4f}")
        logger.info(f"   MAE: ${best_metrics['mae']:,.2f}")
        logger.info(f"   MAPE: {best_metrics['mape']:.2f}%")
        
        logger.info("\n🎉 Pipeline completed successfully!")
        logger.info("=" * 60)
        
        return {
            'dataset': dataset,
            'models': models,
            'results': results_df,
            'cv_results': cv_results,
            'best_model': trainer.best_model,
            'best_model_name': trainer.best_model_name
        }
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()
