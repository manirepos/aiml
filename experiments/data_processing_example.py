#!/usr/bin/env python3
"""
Data Processing Example Script

This script demonstrates common data processing tasks for ML/DL projects.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import sys
import os

# Add src directory to path for importing utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
try:
    from utils import evaluate_model, plot_confusion_matrix
    print("Successfully imported utils module")
except ImportError:
    print("Could not import utils module. Make sure required packages are installed.")


def generate_sample_data():
    """Generate sample classification dataset for demonstration."""
    print("Generating sample dataset...")
    
    # Generate classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"Dataset created with shape: {df.shape}")
    return df


def explore_data(df):
    """Perform basic data exploration."""
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Unique classes: {df['target'].value_counts().to_dict()}")
    
    # Basic statistics
    print("\nDataset info:")
    print(df.info())
    
    # Statistical summary
    print("\nStatistical summary:")
    print(df.describe())
    
    return df


def visualize_data(df):
    """Create visualizations of the dataset."""
    print("\n" + "="*50)
    print("DATA VISUALIZATION")
    print("="*50)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Target distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    df['target'].value_counts().plot(kind='bar')
    plt.title('Target Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Feature correlation heatmap (sample of features)
    plt.subplot(1, 2, 2)
    sample_features = df.select_dtypes(include=[np.number]).columns[:10]
    correlation_matrix = df[sample_features].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlations (Sample)')
    
    plt.tight_layout()
    plt.show()
    
    # Feature distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    feature_cols = [col for col in df.columns if col != 'target'][:6]
    
    for i, feature in enumerate(feature_cols):
        ax = axes[i//3, i%3]
        for target_class in df['target'].unique():
            subset = df[df['target'] == target_class][feature]
            ax.hist(subset, alpha=0.7, label=f'Class {target_class}', bins=20)
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distributions by Class')
    plt.tight_layout()
    plt.show()


def train_model(df):
    """Train a machine learning model."""
    print("\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col != 'target']
    X = df[feature_cols].values
    y = df['target'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Detailed evaluation using utils (if available)
    try:
        metrics = evaluate_model(y_test, y_pred, task_type='classification')
        print("\nDetailed Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    except:
        print("Detailed evaluation not available (utils not imported)")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix (if utils available)
    try:
        plot_confusion_matrix(y_test, y_pred, labels=[f'Class {i}' for i in range(3)])
    except:
        print("Confusion matrix plot not available (utils not imported)")
    
    # Feature importance
    feature_importance = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    top_features = importance_df.head(10)
    plt.bar(range(len(top_features)), top_features['importance'])
    plt.xticks(range(len(top_features)), top_features['feature'], rotation=45)
    plt.title('Top 10 Feature Importance')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    return rf_model, scaler


def main():
    """Main function to run the data processing pipeline."""
    print("Starting ML Data Processing Pipeline")
    print("="*60)
    
    try:
        # Generate sample data
        df = generate_sample_data()
        
        # Explore the data
        df = explore_data(df)
        
        # Visualize the data
        visualize_data(df)
        
        # Train and evaluate model
        model, scaler = train_model(df)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Save results (optional)
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Save the processed data
        output_path = os.path.join(results_dir, 'sample_processed_data.csv')
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
