"""
Quick training script for fraud detection model using a very small subset of data
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the directory to save models
MODEL_DIR = os.path.join('utils', 'ml_models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def main():
    try:
        logger.info("Starting quick fraud detection model training")
        
        # Load a tiny subset of the data for ultra-fast processing
        dataset_path = 'attached_assets/fra.csv'
        logger.info(f"Loading a small subset from {dataset_path}")
        
        # Read only the first 10,000 rows
        df = pd.read_csv(dataset_path, nrows=10000)
        logger.info(f"Loaded {df.shape[0]} rows from the dataset")
        
        # Display basic info
        fraud_count = df['isFraud'].sum()
        logger.info(f"Fraud transactions: {fraud_count} ({fraud_count/df.shape[0]*100:.2f}%)")
        
        # If we have too few fraud cases, oversample them
        if fraud_count < 50:
            logger.info("Oversampling fraud cases to improve model training")
            # Get fraud and non-fraud instances
            fraud_df = df[df['isFraud'] == 1]
            non_fraud_df = df[df['isFraud'] == 0].sample(5000, random_state=42)
            
            # Oversample fraud instances
            fraud_oversampled = fraud_df.sample(200, replace=True, random_state=42)
            
            # Combine oversampled fraud with non-fraud
            df = pd.concat([non_fraud_df, fraud_oversampled])
            logger.info(f"After oversampling: {df.shape[0]} rows with {df['isFraud'].sum()} fraud cases")
        
        # Simple features for quick training
        df['amount_scaled'] = df['amount'] / df['amount'].max()
        
        # Create basic binary features
        df['is_payment'] = (df['type'] == 'PAYMENT').astype(int)
        df['is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
        df['is_cash_out'] = (df['type'] == 'CASH_OUT').astype(int)
        df['is_debit'] = (df['type'] == 'DEBIT').astype(int)
        
        # Select features
        features = ['amount_scaled', 'is_payment', 'is_transfer', 'is_cash_out', 'is_debit']
        X = df[features]
        y = df['isFraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler
        scaler_path = os.path.join(MODEL_DIR, 'fraud_scaler.joblib')
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Train a very simple RF model
        logger.info("Training a simple Random Forest classifier")
        model = RandomForestClassifier(
            n_estimators=20,  # Few trees for quick training
            max_depth=5,      # Shallow trees
            min_samples_split=10,
            random_state=42, 
            class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        logger.info("\nClassification Report:")
        report = classification_report(y_test, y_pred)
        logger.info(f"\n{report}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        logger.info("\nFeature Importance:")
        logger.info(feature_importance)
        
        # Save the model
        model_path = os.path.join(MODEL_DIR, 'fraud_detection_model.joblib')
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        logger.info("Quick fraud detection model successfully trained and saved")
        logger.info("You can restart the application to enable ML-based fraud detection")
        
    except Exception as e:
        logger.error(f"Error in training process: {e}")
        
if __name__ == "__main__":
    main()