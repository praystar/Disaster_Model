import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
import joblib

class ReliefAllocator:
    def __init__(self):
        self.supply_models = {}  # Dictionary to store models for each supply type
        self.label_encoders = {}  # Dictionary to store label encoders for categorical features
        self.feature_columns = [
            'disaster_type',
            'severity',
            'population_density',
            'urban_rural',
            'infrastructure_damage',
            'accessibility',
            'time_since_disaster'
        ]
        
    def _encode_categorical_features(self, df):
        """
        Encode categorical features using LabelEncoder
        """
        encoded_df = df.copy()
        for column in ['disaster_type', 'severity', 'urban_rural', 'accessibility']:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            encoded_df[column] = self.label_encoders[column].fit_transform(df[column])
        return encoded_df
    
    def _prepare_features(self, disaster_data):
        """
        Prepare features for the model
        """
        # Create a DataFrame with the required features
        features = pd.DataFrame({
            'disaster_type': disaster_data['disaster_type'],
            'severity': disaster_data['severity'],
            'population_density': disaster_data.get('population_density', 0),
            'urban_rural': disaster_data.get('urban_rural', 'unknown'),
            'infrastructure_damage': disaster_data.get('infrastructure_damage', 'unknown'),
            'accessibility': disaster_data.get('accessibility', 'unknown'),
            'time_since_disaster': disaster_data.get('time_since_disaster', 0)
        })
        
        # Encode categorical features
        return self._encode_categorical_features(features)
    
    def train(self, training_data):
        """
        Train models for each type of relief supply
        """
        supply_types = ['food', 'water', 'medicine', 'shelter']
        
        for supply_type in supply_types:
            # Prepare features
            X = self._prepare_features(training_data)
            y = training_data[f'{supply_type}_need']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create and train XGBoost model
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Store the trained model
            self.supply_models[supply_type] = model
            
            # Print model performance
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"{supply_type.capitalize()} Model Performance:")
            print(f"Mean Squared Error: {mse:.2f}")
            print(f"Feature Importance:")
            for feature, importance in zip(self.feature_columns, model.feature_importances_):
                print(f"  {feature}: {importance:.4f}")
            print("-" * 50)
    
    def predict_needs(self, disaster_data):
        """
        Predict relief supply needs for a given disaster
        """
        # Prepare features
        X = self._prepare_features(disaster_data)
        
        # Make predictions for each supply type
        predictions = {}
        for supply_type, model in self.supply_models.items():
            predictions[supply_type] = model.predict(X)[0]
        
        return predictions
    
    def save_models(self, directory='models'):
        """
        Save trained models and encoders
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save models
        for supply_type, model in self.supply_models.items():
            model.save_model(f'{directory}/{supply_type}_model.json')
        
        # Save encoders
        for column, encoder in self.label_encoders.items():
            joblib.dump(encoder, f'{directory}/{column}_encoder.joblib')
    
    @classmethod
    def load_models(cls, directory='models'):
        """
        Load trained models and encoders
        """
        instance = cls()
        
        # Load models
        for supply_type in ['food', 'water', 'medicine', 'shelter']:
            model = xgb.XGBRegressor()
            model.load_model(f'{directory}/{supply_type}_model.json')
            instance.supply_models[supply_type] = model
        
        # Load encoders
        for column in ['disaster_type', 'severity', 'urban_rural', 'accessibility']:
            instance.label_encoders[column] = joblib.load(f'{directory}/{column}_encoder.joblib')
        
        return instance 