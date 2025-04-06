"""
Fraud Detection Model for Banking System
This script analyzes financial transaction data to build a machine learning model
for detecting fraudulent transactions. It uses scikit-learn for model building
and joblib for model persistence.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, roc_curve, auc
import joblib
import os

# Set the path for our datasets and models
DATA_PATH = 'attached_assets/fra.csv'
MODEL_PATH = 'models'

# Ensure the model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

def load_and_explore_data(filepath):
    """Load and explore the transaction dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    print("\nDataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nSummary Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nClass Distribution (Fraud vs Non-Fraud):")
    print(df['isFraud'].value_counts())
    print(f"Fraud Percentage: {df['isFraud'].mean() * 100:.4f}%")
    
    return df

def visualize_data(df):
    """Create visualizations for data exploration"""
    # Set up the matplotlib figure
    plt.figure(figsize=(16, 20))
    
    # Distribution of transaction types
    plt.subplot(3, 2, 1)
    sns.countplot(x='type', data=df)
    plt.title('Distribution of Transaction Types')
    plt.xticks(rotation=45)
    
    # Distribution of fraud by transaction type
    plt.subplot(3, 2, 2)
    sns.countplot(x='type', hue='isFraud', data=df)
    plt.title('Fraud Distribution by Transaction Type')
    plt.xticks(rotation=45)
    
    # Amount distribution for fraud and non-fraud transactions
    plt.subplot(3, 2, 3)
    # Use log scale for better visualization
    plt.semilogy()
    sns.boxplot(x='isFraud', y='amount', data=df)
    plt.title('Transaction Amount Distribution by Fraud Status')
    
    # Correlation matrix
    plt.subplot(3, 2, 4)
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Numeric Features')
    
    # Balance before and after transaction for fraud cases
    plt.subplot(3, 2, 5)
    fraud_df = df[df['isFraud'] == 1]
    
    # Create bar plot for origin account balance
    indices = range(len(fraud_df))
    width = 0.35
    plt.bar(indices, fraud_df['oldbalanceOrg'], width, label='Original Balance')
    plt.bar([i + width for i in indices], fraud_df['newbalanceOrig'], width, label='New Balance')
    plt.title('Account Balance Before and After Fraudulent Transactions')
    plt.xlabel('Transaction Index')
    plt.ylabel('Balance')
    plt.legend()
    
    # Save the visualizations
    plt.tight_layout()
    plt.savefig('fraud_analysis_visualizations.png')
    print("Visualizations saved to 'fraud_analysis_visualizations.png'")

def preprocess_data(df):
    """Preprocess the data for model training"""
    print("\nPreprocessing data...")
    
    # Convert categorical variables to dummy variables
    df_processed = pd.get_dummies(df, columns=['type'], drop_first=True)
    
    # Drop unnecessary columns
    cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'step']
    df_processed = df_processed.drop(cols_to_drop, axis=1)
    
    # Split features and target
    X = df_processed.drop('isFraud', axis=1)
    y = df_processed['isFraud']
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Save the scaler for future use
    joblib.dump(scaler, f'{MODEL_PATH}/fraud_scaler.joblib')
    
    return X, y

def train_model(X, y):
    """Train a machine learning model"""
    print("\nSplitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    print("Training Random Forest model...")
    # Initialize and train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('precision_recall_curve.png')
    
    # Feature Importance
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15))
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    
    print("\nEvaluation plots saved to 'roc_curve.png', 'precision_recall_curve.png', and 'feature_importance.png'")
    
    # Save the model
    joblib.dump(model, f'{MODEL_PATH}/fraud_detection_model.joblib')
    
    # Save feature names for future prediction
    joblib.dump(list(X.columns), f'{MODEL_PATH}/feature_names.joblib')
    
    print(f"\nModel saved to '{MODEL_PATH}/fraud_detection_model.joblib'")
    
    return model, feature_importances

def main():
    """Main function to run the complete analysis and modeling pipeline"""
    print("=== Fraud Detection Model Development ===")
    
    # Load and explore data
    df = load_and_explore_data(DATA_PATH)
    
    # Visualize data
    visualize_data(df)
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Train and evaluate model
    model, feature_importances = train_model(X, y)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importances.head(10))
    
    print("\n=== Fraud Detection Model Development Complete ===")
    
if __name__ == "__main__":
    main()