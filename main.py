"""
Main Execution Script for EV Range Prediction Project
Orchestrates the entire pipeline from data loading to prediction
"""

import os
import sys

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import EVDataPreprocessor
from model_training import EVModelTrainer
from predict import EVRangePredictor


def main():
    """Main pipeline execution"""
    
    print("\n" + "="*70)
    print(" " * 15 + "EV RANGE PREDICTION PROJECT")
    print(" " * 20 + "Complete Pipeline")
    print("="*70)
    
    # Step 1: Data Preprocessing
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    preprocessor = EVDataPreprocessor(data_path='data/electric_vehicle_analytics.csv')
    preprocessor.load_data()
    preprocessor.check_data_quality()
    preprocessor.get_feature_info()
    
    X, y = preprocessor.prepare_features()
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Step 2: Model Training
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)
    
    trainer = EVModelTrainer()
    trainer.initialize_models()
    trainer.train_models(X_train, X_test, y_train, y_test)
    trainer.display_comparison()
    trainer.select_best_model()
    
    # Save model and results
    trainer.save_model()
    trainer.save_results()
    trainer.get_feature_importance(X.columns.tolist())
    
    # Step 3: Making Predictions
    print("\n" + "="*70)
    print("STEP 3: MAKING PREDICTIONS")
    print("="*70)
    
    predictor = EVRangePredictor()
    predictor.load_model()
    
    # Example predictions
    print("\n📊 Example Predictions:")
    print("-" * 70)
    
    examples = [
        {
            'name': 'Tesla Model 3',
            'params': {
                'battery_capacity': 75.0,
                'battery_health': 95.0,
                'charging_power': 150.0,
                'charging_time': 0.5,
                'charge_cycles': 500,
                'energy_consumption': 16.5,
                'mileage': 50000,
                'avg_speed': 65.0,
                'max_speed': 200,
                'acceleration': 5.5,
                'temperature': 20.0
            }
        },
        {
            'name': 'Nissan Leaf',
            'params': {
                'battery_capacity': 40.0,
                'battery_health': 90.0,
                'charging_power': 50.0,
                'charging_time': 1.0,
                'charge_cycles': 800,
                'energy_consumption': 18.0,
                'mileage': 80000,
                'avg_speed': 55.0,
                'max_speed': 150,
                'acceleration': 7.5,
                'temperature': 15.0
            }
        },
        {
            'name': 'BMW i4',
            'params': {
                'battery_capacity': 80.0,
                'battery_health': 97.0,
                'charging_power': 200.0,
                'charging_time': 0.4,
                'charge_cycles': 300,
                'energy_consumption': 17.0,
                'mileage': 30000,
                'avg_speed': 70.0,
                'max_speed': 220,
                'acceleration': 4.5,
                'temperature': 22.0
            }
        }
    ]
    
    for example in examples:
        print(f"\n🚗 {example['name']}:")
        predicted_range = predictor.predict(**example['params'])
        print(f"   Predicted Range: {predicted_range} km")
    
    # Final Summary
    print("\n" + "="*70)
    print("PROJECT EXECUTION SUMMARY")
    print("="*70)
    print("\n✓ Data preprocessing completed")
    print("✓ Models trained and evaluated")
    print("✓ Best model saved")
    print("✓ Predictions generated")
    print("\n📁 Output Files:")
    print("   - models/ev_range_predictor.pkl")
    print("   - results/model_results.json")
    print("   - results/feature_importance.csv")
    
    print("\n" + "="*70)
    print(" " * 20 + "PROJECT COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()