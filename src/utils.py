"""
ML/DL Utilities Module

This module contains utility functions for machine learning and deep learning projects.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Args:
        file_path: Path to the data file
        **kwargs: Additional arguments for pandas read functions
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, **kwargs)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path, **kwargs)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def split_data(X: np.ndarray, y: np.ndarray, 
               test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def plot_learning_curve(train_scores: List[float], 
                       val_scores: List[float],
                       title: str = "Learning Curve") -> None:
    """
    Plot training and validation learning curves.
    
    Args:
        train_scores: Training scores over epochs
        val_scores: Validation scores over epochs
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_scores) + 1)
    
    plt.plot(epochs, train_scores, 'b-', label='Training Score')
    plt.plot(epochs, val_scores, 'r-', label='Validation Score')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                  task_type: str = 'classification') -> dict:
    """
    Evaluate model performance with appropriate metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        task_type: 'classification' or 'regression'
        
    Returns:
        Dictionary of evaluation metrics
    """
    if task_type == 'classification':
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
    elif task_type == 'regression':
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         labels: Optional[List[str]] = None) -> None:
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
