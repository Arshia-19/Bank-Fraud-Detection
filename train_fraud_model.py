"""
Script to train the fraud detection model using the provided CSV file
and integrate it with our banking system
"""
import os
import sys
from fraud_detection_model import load_and_explore_data, preprocess_data, train_model

def main():
    """Train the fraud detection model and save it for system integration"""
    print("=== Training Fraud Detection Model ===")
    
    # Check if the data file exists
    data_path = 'attached_assets/fra.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    # Load and explore the data
    print(f"Loading data from {data_path}...")
    df = load_and_explore_data(data_path)
    
    # Preprocess the data
    X, y = preprocess_data(df)
    
    # Train the model
    model, feature_importances = train_model(X, y)
    
    print("\nFraud Detection Model training complete!")
    print("The model has been saved and is ready for integration with the banking system.")
    print("Features used by the model in order of importance:")
    print(feature_importances.head(10))
    
    print("\nTo use this model for fraud detection:")
    print("1. Restart the banking application")
    print("2. The system will automatically use the model for transaction fraud detection")
    
if __name__ == "__main__":
    main()