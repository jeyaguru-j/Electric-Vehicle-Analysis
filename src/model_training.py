"""
Model Training Module for EV Range Prediction
Trains multiple ML models and selects the best performer
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import EVDataPreprocessor


class EVModelTrainer:
    """Class to handle model training and evaluation"""
    
    def __init__(self):
        """Initialize the model trainer"""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize different regression models"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=5,
                learning_rate=0.1
            )
        }
        print(f"✓ Initialized {len(self.models)} models")
        return self
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train all models and evaluate performance
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
        """
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mse = mean_squared_error(y_test, y_pred_test)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Store results
            self.results[name] = {
                'model': model,
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'mse': float(test_mse),
                'rmse': float(test_rmse),
                'mae': float(test_mae),
                'predictions': y_pred_test
            }
            
            # Print results
            print(f"  ✓ Training R² Score: {train_r2:.4f}")
            print(f"  ✓ Testing R² Score: {test_r2:.4f}")
            print(f"  ✓ RMSE: {test_rmse:.2f} km")
            print(f"  ✓ MAE: {test_mae:.2f} km")
        
        return self
    
    def display_comparison(self):
        """Display comparison of all models"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        print(f"{'Model':<20} {'Train R²':<12} {'Test R²':<12} {'RMSE (km)':<12} {'MAE (km)':<12}")
        print("-"*68)
        
        for name, metrics in self.results.items():
            print(f"{name:<20} {metrics['train_r2']:<12.4f} {metrics['test_r2']:<12.4f} "
                  f"{metrics['rmse']:<12.2f} {metrics['mae']:<12.2f}")
    
    def select_best_model(self):
        """Select the best model based on test R² score"""
        best_r2 = -float('inf')
        
        for name, metrics in self.results.items():
            if metrics['test_r2'] > best_r2:
                best_r2 = metrics['test_r2']
                self.best_model_name = name
                self.best_model = metrics['model']
        
        print(f"\n✓ Best Model: {self.best_model_name}")
        print(f"  - Test R² Score: {self.results[self.best_model_name]['test_r2']:.4f}")
        print(f"  - RMSE: {self.results[self.best_model_name]['rmse']:.2f} km")
        
        return self
    
    def save_model(self, output_path='models/ev_range_predictor.pkl'):
        """Save the best model to disk"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self.best_model, output_path)
        print(f"\n✓ Best model saved to: {output_path}")
        return self
    
    def save_results(self, output_path='results/model_results.json'):
        """Save model results to JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare results for JSON (remove non-serializable objects)
        results_for_json = {}
        for name, metrics in self.results.items():
            results_for_json[name] = {
                'train_r2': metrics['train_r2'],
                'test_r2': metrics['test_r2'],
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae']
            }
        
        with open(output_path, 'w') as f:
            json.dump(results_for_json, f, indent=4)
        
        print(f"✓ Results saved to: {output_path}")
        return self
    
    def get_feature_importance(self, feature_names, output_path='results/feature_importance.csv'):
        """Get and save feature importance for tree-based models"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': self.best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            importance_df.to_csv(output_path, index=False)
            
            print(f"\n✓ Feature importance saved to: {output_path}")
            print("\nTop 5 Most Important Features:")
            for idx, row in importance_df.head().iterrows():
                print(f"  {row['Feature']}: {row['Importance']:.4f}")
        else:
            print("\n⚠ Feature importance not available for this model type")
        
        return self


def main():
    """Main function to train models"""
    print("="*60)
    print("EV RANGE PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Preprocess data
    preprocessor = EVDataPreprocessor()
    preprocessor.load_data()
    X, y = preprocessor.prepare_features()
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Train models
    trainer = EVModelTrainer()
    trainer.initialize_models()
    trainer.train_models(X_train, X_test, y_train, y_test)
    trainer.display_comparison()
    trainer.select_best_model()
    
    # Save model and results
    trainer.save_model()
    trainer.save_results()
    trainer.get_feature_importance(X.columns.tolist())
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()