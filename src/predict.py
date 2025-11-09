"""
Prediction Module for EV Range Prediction
Load trained model and make predictions
"""

import pandas as pd
import numpy as np
import joblib
import os


class EVRangePredictor:
    """Class to handle EV range predictions"""
    
    def __init__(self, model_path='models/ev_range_predictor.pkl'):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the saved model
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = [
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
    
    def load_model(self):
        """Load the trained model from disk"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        print(f"✓ Model loaded successfully from: {self.model_path}")
        return self
    
    def predict(self, battery_capacity, battery_health, charging_power,
                charging_time, charge_cycles, energy_consumption,
                mileage, avg_speed, max_speed, acceleration, temperature):
        """
        Predict EV range based on vehicle parameters
        
        Args:
            battery_capacity: Battery capacity in kWh
            battery_health: Battery health percentage
            charging_power: Charging power in kW
            charging_time: Charging time in hours
            charge_cycles: Number of charge cycles
            energy_consumption: Energy consumption in kWh per 100km
            mileage: Total mileage in km
            avg_speed: Average speed in km/h
            max_speed: Maximum speed in km/h
            acceleration: 0-100 km/h acceleration time in seconds
            temperature: Ambient temperature in Celsius
            
        Returns:
            Predicted range in kilometers
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Create feature array
        features = np.array([[
            battery_capacity, battery_health, charging_power,
            charging_time, charge_cycles, energy_consumption,
            mileage, avg_speed, max_speed, acceleration, temperature
        ]])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        return round(prediction, 2)
    
    def predict_from_dataframe(self, df):
        """
        Predict EV range for multiple vehicles from a dataframe
        
        Args:
            df: DataFrame with required features
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select features in correct order
        X = df[self.feature_names]
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_and_explain(self, battery_capacity, battery_health, charging_power,
                           charging_time, charge_cycles, energy_consumption,
                           mileage, avg_speed, max_speed, acceleration, temperature):
        """
        Predict EV range and provide explanation
        
        Returns:
            Dictionary with prediction and input summary
        """
        prediction = self.predict(
            battery_capacity, battery_health, charging_power,
            charging_time, charge_cycles, energy_consumption,
            mileage, avg_speed, max_speed, acceleration, temperature
        )
        
        result = {
            'predicted_range_km': prediction,
            'input_parameters': {
                'Battery Capacity (kWh)': battery_capacity,
                'Battery Health (%)': battery_health,
                'Charging Power (kW)': charging_power,
                'Charging Time (hr)': charging_time,
                'Charge Cycles': charge_cycles,
                'Energy Consumption (kWh/100km)': energy_consumption,
                'Mileage (km)': mileage,
                'Average Speed (km/h)': avg_speed,
                'Max Speed (km/h)': max_speed,
                'Acceleration 0-100 (sec)': acceleration,
                'Temperature (°C)': temperature
            }
        }
        
        return result


def main():
    """Main function to demonstrate prediction"""
    print("="*60)
    print("EV RANGE PREDICTION - MAKING PREDICTIONS")
    print("="*60)
    
    # Initialize predictor
    predictor = EVRangePredictor()
    predictor.load_model()
    
    # Example 1: Tesla Model 3 (typical specifications)
    print("\n📊 Example 1: Tesla Model 3")
    print("-" * 60)
    result = predictor.predict_and_explain(
        battery_capacity=75.0,
        battery_health=95.0,
        charging_power=150.0,
        charging_time=0.5,
        charge_cycles=500,
        energy_consumption=16.5,
        mileage=50000,
        avg_speed=65.0,
        max_speed=200,
        acceleration=5.5,
        temperature=20.0
    )
    
    print(f"\n✓ Predicted Range: {result['predicted_range_km']} km")
    
    # Example 2: Nissan Leaf
    print("\n📊 Example 2: Nissan Leaf")
    print("-" * 60)
    predicted_range = predictor.predict(
        battery_capacity=40.0,
        battery_health=90.0,
        charging_power=50.0,
        charging_time=1.0,
        charge_cycles=800,
        energy_consumption=18.0,
        mileage=80000,
        avg_speed=55.0,
        max_speed=150,
        acceleration=7.5,
        temperature=15.0
    )
    
    print(f"\n✓ Predicted Range: {predicted_range} km")
    
    # Example 3: High-performance EV
    print("\n📊 Example 3: High-Performance EV")
    print("-" * 60)
    predicted_range = predictor.predict(
        battery_capacity=100.0,
        battery_health=98.0,
        charging_power=250.0,
        charging_time=0.3,
        charge_cycles=200,
        energy_consumption=20.0,
        mileage=15000,
        avg_speed=80.0,
        max_speed=240,
        acceleration=3.5,
        temperature=25.0
    )
    
    print(f"\n✓ Predicted Range: {predicted_range} km")
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()