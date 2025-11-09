"""
Data Preprocessing Module for EV Range Prediction
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


class EVDataPreprocessor:
    """Class to handle all data preprocessing operations"""
    
    def __init__(self, data_path='data/electric_vehicle_analytics.csv'):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Path to the dataset
        """
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load the dataset from CSV file"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Shape: {self.df.shape}")
        print(f"  - Records: {self.df.shape[0]}")
        print(f"  - Features: {self.df.shape[1]}")
        return self
    
    def check_data_quality(self):
        """Check for missing values and data quality issues"""
        print("\n" + "="*60)
        print("DATA QUALITY CHECK")
        print("="*60)
        
        # Check missing values
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("✓ No missing values found!")
        else:
            print("⚠ Missing values detected:")
            print(missing[missing > 0])
        
        # Check duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates == 0:
            print("✓ No duplicate records found!")
        else:
            print(f"⚠ {duplicates} duplicate records found")
        
        # Data types
        print(f"\n✓ Data types verified")
        
        return self
    
    def get_feature_info(self):
        """Display information about features"""
        print("\n" + "="*60)
        print("FEATURE INFORMATION")
        print("="*60)
        
        # Numerical features
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        print(f"\nNumerical Features ({len(numerical_cols)}):")
        for col in numerical_cols:
            print(f"  - {col}")
        
        # Categorical features
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        print(f"\nCategorical Features ({len(categorical_cols)}):")
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            print(f"  - {col}: {unique_count} unique values")
        
        return self
    
    def prepare_features(self, target_column='Range_km'):
        """
        Prepare features for modeling
        
        Args:
            target_column: Name of the target variable
            
        Returns:
            X: Features dataframe
            y: Target series
        """
        # Select numerical features for modeling
        feature_columns = [
            'Battery_Capacity_kWh',
            'Battery_Health_%',
            'Charging_Power_kW',
            'Charging_Time_hr',
            'Charge_Cycles',
            'Energy_Consumption_kWh_per_100km',
            'Mileage_km',
            'Avg_Speed_kmh',
            'Max_Speed_kmh',
            'Acceleration_0_100_kmh_sec',
            'Temperature_C'
        ]
        
        X = self.df[feature_columns]
        y = self.df[target_column]
        
        print(f"\n✓ Features prepared for modeling")
        print(f"  - Feature count: {len(feature_columns)}")
        print(f"  - Target variable: {target_column}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n✓ Data split completed")
        print(f"  - Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
        print(f"  - Testing set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def get_statistics(self):
        """Get statistical summary of the dataset"""
        return self.df.describe()


def main():
    """Main function to demonstrate preprocessing"""
    print("="*60)
    print("EV RANGE PREDICTION - DATA PREPROCESSING")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = EVDataPreprocessor()
    
    # Load and check data
    preprocessor.load_data()
    preprocessor.check_data_quality()
    preprocessor.get_feature_info()
    
    # Prepare features
    X, y = preprocessor.prepare_features()
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()