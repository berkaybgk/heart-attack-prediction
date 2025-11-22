import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


def load_data(train_path, test_path):
    """Load preprocessed training and test data."""
    print("Loading preprocessed data...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"Training data shape: {df_train.shape}")
    print(f"Test data shape: {df_test.shape}")
    
    return df_train, df_test


def prepare_features_target(df_train, df_test, target_column='HadHeartAttack'):
    """Separate features and target variable."""
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]
    
    print(f"\nFeature columns: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"\nTraining class distribution:\n{y_train.value_counts()}")
    print(f"\nTest class distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, **kwargs):
    """
    Train a Random Forest classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    **kwargs : dict
        Additional parameters for RandomForestClassifier
    """
    print("\n" + "="*60)
    print("Training Random Forest Classifier...")
    print("="*60)
    
    # Default hyperparameters (can be overridden by kwargs)
    default_params = {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 1,
        'class_weight': 'balanced'
    }
    
    # Update with any provided kwargs
    default_params.update(kwargs)
    
    print("\nModel Parameters:")
    for key, value in default_params.items():
        print(f"  {key}: {value}")
    
    # Initialize and train the model
    rf_model = RandomForestClassifier(**default_params)
    
    start_time = datetime.now()
    rf_model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    return rf_model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate the model on both training and test sets.
    
    Returns:
    --------
    dict : Dictionary containing all evaluation metrics
    """
    print("\n" + "="*60)
    print("Evaluating Model Performance...")
    print("="*60)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Get probability predictions for ROC-AUC
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics for training set
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1': f1_score(y_train, y_train_pred),
        'roc_auc': roc_auc_score(y_train, y_train_proba)
    }
    
    # Calculate metrics for test set
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba)
    }
    
    # Print metrics
    print("\n" + "-"*60)
    print("TRAINING SET METRICS:")
    print("-"*60)
    for metric, value in train_metrics.items():
        print(f"{metric.upper():15s}: {value:.4f}")
    
    print("\n" + "-"*60)
    print("TEST SET METRICS:")
    print("-"*60)
    for metric, value in test_metrics.items():
        print(f"{metric.upper():15s}: {value:.4f}")
    
    # Print classification report for test set
    print("\n" + "-"*60)
    print("DETAILED CLASSIFICATION REPORT (Test Set):")
    print("-"*60)
    print(classification_report(y_test, y_test_pred, target_names=['No Heart Attack', 'Heart Attack']))
    
    # Confusion matrices
    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    print("\n" + "-"*60)
    print("CONFUSION MATRIX (Test Set):")
    print("-"*60)
    print(f"True Negatives:  {test_cm[0, 0]:6d}  |  False Positives: {test_cm[0, 1]:6d}")
    print(f"False Negatives: {test_cm[1, 0]:6d}  |  True Positives:  {test_cm[1, 1]:6d}")
    
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'train_cm': train_cm,
        'test_cm': test_cm,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'y_train_proba': y_train_proba,
        'y_test_proba': y_test_proba
    }


def plot_confusion_matrix(cm, title, save_path=None):
    """Plot confusion matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Heart Attack', 'Heart Attack'],
                yticklabels=['No Heart Attack', 'Heart Attack'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_proba, title, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
    
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Plot feature importance from the Random Forest model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top N features
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_importances[::-1], align='center', color='steelblue')
    plt.yticks(range(top_n), [top_features[i] for i in range(top_n-1, -1, -1)])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")
    
    plt.show()
    
    # Print feature importances
    print("\n" + "-"*60)
    print(f"TOP {top_n} FEATURE IMPORTANCES:")
    print("-"*60)
    for i, (feature, importance) in enumerate(zip(top_features, top_importances), 1):
        print(f"{i:2d}. {feature:50s}: {importance:.6f}")


def save_model(model, save_path):
    """Save the trained model to disk."""
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {save_path}")


def save_results_summary(model, evaluation_results, save_path):
    """Save a summary of results to a text file."""
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("RANDOM FOREST MODEL - HEART ATTACK PREDICTION\n")
        f.write("="*60 + "\n")
        f.write(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("MODEL PARAMETERS:\n")
        f.write("-"*60 + "\n")
        for key, value in model.get_params().items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("TRAINING SET METRICS:\n")
        f.write("-"*60 + "\n")
        for metric, value in evaluation_results['train_metrics'].items():
            f.write(f"{metric.upper():15s}: {value:.4f}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("TEST SET METRICS:\n")
        f.write("-"*60 + "\n")
        for metric, value in evaluation_results['test_metrics'].items():
            f.write(f"{metric.upper():15s}: {value:.4f}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("CONFUSION MATRIX (Test Set):\n")
        f.write("-"*60 + "\n")
        cm = evaluation_results['test_cm']
        f.write(f"True Negatives:  {cm[0, 0]:6d}  |  False Positives: {cm[0, 1]:6d}\n")
        f.write(f"False Negatives: {cm[1, 0]:6d}  |  True Positives:  {cm[1, 1]:6d}\n")
    
    print(f"Results summary saved to {save_path}")


def main():
    """Main execution function."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define paths
    TRAIN_PATH = '../resources/heart_2022_train_processed.csv'
    TEST_PATH = '../resources/heart_2022_test_processed.csv'
    MODEL_SAVE_PATH = '../models/random_forest_model.pkl'
    RESULTS_DIR = '../results'
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # Load data
    df_train, df_test = load_data(TRAIN_PATH, TEST_PATH)
    
    # Prepare features and target
    X_train, X_test, y_train, y_test = prepare_features_target(df_train, df_test)
    
    # Train Random Forest model
    rf_model = train_random_forest(
        X_train, 
        y_train,
        n_estimators=300,            # Number of trees
        max_depth=30,                # Maximum depth of trees
        min_samples_split=10,        # Minimum samples to split a node
        min_samples_leaf=4,          # Minimum samples in a leaf
        max_features='sqrt',         # Number of features to consider for best split
        random_state=42,
        n_jobs=-1,                   # Use all CPU cores
        class_weight='balanced'      # Handle class imbalance
    )
    
    # Evaluate model
    evaluation_results = evaluate_model(rf_model, X_train, X_test, y_train, y_test)
    
    # Plot confusion matrix for test set
    plot_confusion_matrix(
        evaluation_results['test_cm'],
        'Confusion Matrix - Test Set',
        save_path=os.path.join(RESULTS_DIR, 'confusion_matrix_test.png')
    )
    
    # Plot ROC curve for test set
    plot_roc_curve(
        y_test,
        evaluation_results['y_test_proba'],
        'ROC Curve - Test Set',
        save_path=os.path.join(RESULTS_DIR, 'roc_curve_test.png')
    )
    
    # Plot feature importance
    plot_feature_importance(
        rf_model,
        X_train.columns.tolist(),
        top_n=20,
        save_path=os.path.join(RESULTS_DIR, 'feature_importance.png')
    )
    
    # Save model
    save_model(rf_model, MODEL_SAVE_PATH)
    
    # Save results summary
    save_results_summary(
        rf_model,
        evaluation_results,
        os.path.join(RESULTS_DIR, 'results_summary.txt')
    )
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Results saved to: {RESULTS_DIR}/")
    
    return rf_model, evaluation_results


if __name__ == "__main__":
    model, results = main()

