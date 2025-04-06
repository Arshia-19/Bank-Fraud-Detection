"""
Script to update the transaction_type column in the transactions table to VARCHAR(30)
"""
from main import app
from app import db
from sqlalchemy import text

def update_transaction_type_length():
    """Updates the transaction_type column in the transactions table to VARCHAR(30)"""
    with app.app_context():
        # Execute SQL to alter the column
        try:
            db.session.execute(text('''
                ALTER TABLE transactions 
                ALTER COLUMN transaction_type TYPE VARCHAR(30);
            '''))
            db.session.commit()
            print("Successfully updated transaction_type column to VARCHAR(30)")
        except Exception as e:
            db.session.rollback()
            print(f"Error updating column: {e}")
        
if __name__ == "__main__":
    update_transaction_type_length()