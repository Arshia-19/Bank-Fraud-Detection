"""
Optimized script to train the fraud detection model using a subset of the provided CSV file
and integrate it with our banking system
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
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
        logger.info("Starting optimized fraud detection model training process")
        
        # Check if dataset file exists
        dataset_path = 'attached_assets/fra.csv'
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found at {dataset_path}")
            return
        
        # Load a subset of the data for faster processing
        logger.info(f"Loading a subset of data from {dataset_path}")
        
        # Read only the first 50,000 rows (adjust as needed based on memory constraints)
        df = pd.read_csv(dataset_path, nrows=50000)
        logger.info(f"Loaded {df.shape[0]} rows from the dataset")
        
        # Display basic dataset info
        logger.info(f"Dataset columns: {df.columns.tolist()}")
        logger.info(f"Fraud transactions: {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.2f}%)")
        
        # Create new features
        df['transactionAmount'] = df['amount']
        df['oldBalanceDiff'] = df['oldbalanceOrg'] - df['oldbalanceDest']
        df['newBalanceDiff'] = df['newbalanceOrig'] - df['newbalanceDest']
        df['balanceChange'] = (df['newbalanceOrig'] - df['oldbalanceOrg']) + (df['newbalanceDest'] - df['oldbalanceDest'])
        
        # Convert 'type' to numerical using one-hot encoding
        type_dummies = pd.get_dummies(df['type'], prefix='type')
        df = pd.concat([df, type_dummies], axis=1)
        
        # Select features
        features = ['transactionAmount', 'oldBalanceDiff', 'newBalanceDiff', 'balanceChange']
        
        # Add transaction type features if they exist
        for feature in ['type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']:
            if feature in df.columns:
                features.append(feature)
            else:
                logger.warning(f"Feature {feature} not found in dataset, skipping")
        
        # Create X (features) and y (target)
        X = df[features]
        y = df['isFraud']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler for later use
        scaler_path = os.path.join(MODEL_DIR, 'fraud_scaler.joblib')
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Train a lightweight Random Forest model (fewer trees)
        logger.info("Training Random Forest classifier with fewer trees for faster processing")
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test_scaled)
        
        # Print classification report
        logger.info("\nClassification Report:")
        report = classification_report(y_test, y_pred)
        logger.info(f"\n{report}")
        
        # Compute ROC curve and AUC
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        logger.info(f"AUC: {roc_auc:.4f}")
        
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
        
        logger.info("Fraud detection model successfully trained and saved")
        logger.info("You can restart the application to enable ML-based fraud detection")
        
    except Exception as e:
        logger.error(f"Error in training process: {e}")
        
if __name__ == "__main__":
    main()