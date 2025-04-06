"""
Script to test the fraud detection model with sample transactions
"""

import logging
from utils.ml_fraud_detection import check_ml_fraud, get_model_info

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the fraud detection model with sample transactions"""
    logger.info("Testing fraud detection model with sample transactions")
    
    # Get model information
    model_info = get_model_info()
    logger.info(f"\nModel Information:\n{model_info}")
    
    # Test cases - transaction_id, user_id, recipient_id, amount, transaction_type
    test_cases = [
        # Normal cases
        (1, 100, 200, 5000, "transfer"),             # Small transfer
        (2, 101, 201, 2500, "payment"),              # Small payment
        (3, 102, None, 3000, "withdrawal"),          # Small withdrawal
        
        # Potentially fraudulent cases
        (4, 103, 202, 500000, "transfer"),           # Large transfer
        (5, 104, None, 1000000, "withdrawal"),       # Very large withdrawal
        (6, 105, 203, 250000, "international_transfer"),  # International transfer
        
        # Edge cases
        (7, 106, 204, 10, "transfer"),               # Tiny transfer
        (8, 107, None, 9999999, "withdrawal"),       # Extreme withdrawal
    ]
    
    # Run each test case
    for case in test_cases:
        transaction_id, user_id, recipient_id, amount, transaction_type = case
        
        # Check for fraud
        is_fraud, reason = check_ml_fraud(
            transaction_id=transaction_id,
            user_id=user_id,
            recipient_id=recipient_id,
            amount=amount,
            transaction_type=transaction_type
        )
        
        # Log the results
        logger.info(f"\nTransaction {transaction_id}: ₹{amount:.2f} ({transaction_type})")
        if is_fraud:
            logger.warning(f"FRAUD DETECTED: {reason}")
        else:
            logger.info("Result: No fraud detected")
    
if __name__ == "__main__":
    main()