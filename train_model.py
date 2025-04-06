"""
Script to train the fraud detection model using the provided CSV file
and integrate it with our banking system
"""

import os
import sys
import logging
import traceback
from utils.train_fraud_model import load_and_preprocess_data, train_model, save_model

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Train the fraud detection model and save it for system integration"""
    try:
        logger.info("Starting fraud detection model training process")
        
        # Check if dataset file exists
        dataset_path = 'attached_assets/fra.csv'
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found at {dataset_path}")
            return
        
        # Load and preprocess data
        logger.info(f"Loading data from {dataset_path}")
        X, y = load_and_preprocess_data(dataset_path)
        
        if X is None or y is None:
            logger.error("Failed to load or preprocess data")
            return
        
        # Train the model
        logger.info("Training fraud detection model")
        model, scaler = train_model(X, y)
        
        # Save the model
        logger.info("Saving model for use in the banking system")
        model_path = save_model(model)
        
        logger.info(f"Fraud detection model successfully trained and saved to {model_path}")
        logger.info("The model is now ready to be used by the banking system")
        logger.info("You can restart the application to enable ML-based fraud detection")
        
    except Exception as e:
        logger.error(f"Error training fraud detection model: {e}")
        logger.error(traceback.format_exc())
        
if __name__ == "__main__":
    main()