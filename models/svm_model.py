import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
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
    
    # Calculate class weights
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    class_weight_ratio = neg_count / pos_count
    print(f"\nClass imbalance ratio (neg/pos): {class_weight_ratio:.2f}")
    
    return X_train, X_test, y_train, y_test


def train_svm(X_train, y_train, kernel='rbf', use_calibration=True, **kwargs):
    """
    Train an SVM classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    kernel : str
        Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
    use_calibration : bool
        Whether to calibrate probabilities using CalibratedClassifierCV
    **kwargs : dict
        Additional parameters for SVC
    """
    print("\n" + "="*60)
    print("Training SVM Classifier...")
    print("="*60)
    
    # Default hyperparameters (can be overridden by kwargs)
    default_params = {
        'kernel': kernel,
        'C': 1.0,                    # Regularization parameter
        'gamma': 'scale',            # Kernel coefficient
        'class_weight': 'balanced',  # Handle class imbalance
        'random_state': 42,
        'max_iter': -1,              # No limit on iterations
        'cache_size': 2000,          # Larger cache for faster training
        'verbose': True
    }
    
    # For probability estimates, we need probability=True
    # But this slows down training significantly for large datasets
    if not use_calibration:
        default_params['probability'] = True
    
    # Update with any provided kwargs
    default_params.update(kwargs)
    
    print("\nModel Parameters:")
    for key, value in default_params.items():
        print(f"  {key}: {value}")
    
    # Initialize the base SVM model
    svm_model = SVC(**default_params)
    
    start_time = datetime.now()
    
    if use_calibration:
        print("\n" + "-"*60)
        print("Training with probability calibration...")
        print("This may take longer but provides better probability estimates.")
        print("-"*60)
        
        # Train base model and calibrate probabilities
        svm_model.fit(X_train, y_train)
        
        # Calibrate probabilities using 5-fold cross-validation
        print("\nCalibrating probabilities...")
        calibrated_model = CalibratedClassifierCV(
            svm_model, 
            method='sigmoid',  # Platt scaling
            cv=5,
            n_jobs=-1
        )
        calibrated_model.fit(X_train, y_train)
        final_model = calibrated_model
    else:
        # Train directly with probability=True
        svm_model.fit(X_train, y_train)
        final_model = svm_model
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return final_model


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
    print("Making predictions on training set...")
    y_train_pred = model.predict(X_train)
    print("Making predictions on test set...")
    y_test_pred = model.predict(X_test)
    
    # Get probability predictions for ROC-AUC
    print("Computing probability estimates...")
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


def plot_support_vectors_summary(model, save_path=None):
    """Plot summary information about support vectors."""
    # Get the base model (unwrap from CalibratedClassifierCV if needed)
    if hasattr(model, 'calibrated_classifiers_'):
        base_models = [clf.estimator for clf in model.calibrated_classifiers_]
        n_support_list = [clf.estimator.n_support_ for clf in model.calibrated_classifiers_]
        avg_n_support = np.mean([sum(ns) for ns in n_support_list])
        
        print("\n" + "-"*60)
        print("SUPPORT VECTORS SUMMARY (Calibrated Model):")
        print("-"*60)
        print(f"Number of calibrated models: {len(base_models)}")
        print(f"Average support vectors across folds: {avg_n_support:.0f}")
    else:
        n_support = model.n_support_
        print("\n" + "-"*60)
        print("SUPPORT VECTORS SUMMARY:")
        print("-"*60)
        print(f"Support vectors (class 0): {n_support[0]}")
        print(f"Support vectors (class 1): {n_support[1]}")
        print(f"Total support vectors: {sum(n_support)}")
        print(f"Percentage of training data: {sum(n_support)/model.n_support_.sum()*100:.2f}%")


def save_model(model, save_path):
    """Save the trained model to disk."""
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {save_path}")


def save_results_summary(model, evaluation_results, save_path):
    """Save a summary of results to a text file."""
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SVM MODEL - HEART ATTACK PREDICTION\n")
        f.write("="*60 + "\n")
        f.write(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("MODEL PARAMETERS:\n")
        f.write("-"*60 + "\n")
        
        # Handle calibrated models
        if hasattr(model, 'calibrated_classifiers_'):
            f.write("Model Type: Calibrated SVM (CalibratedClassifierCV)\n")
            base_model = model.calibrated_classifiers_[0].estimator
            for key, value in base_model.get_params().items():
                f.write(f"{key}: {value}\n")
        else:
            f.write("Model Type: Standard SVM\n")
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
    MODEL_SAVE_PATH = '../models/svm_model.pkl'
    RESULTS_DIR = '../results/svm'
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # Load data
    df_train, df_test = load_data(TRAIN_PATH, TEST_PATH)
    
    # Prepare features and target
    X_train, X_test, y_train, y_test = prepare_features_target(df_train, df_test)
    
    print("\n" + "="*60)
    print("IMPORTANT NOTE:")
    print("="*60)
    print("SVM training on large datasets can be very slow.")
    print("For datasets with 300k+ samples, consider:")
    print("  1. Using a smaller sample for initial experiments")
    print("  2. Using LinearSVC or SGDClassifier for faster training")
    print("  3. Using a linear kernel instead of RBF")
    print("="*60)
    
    # Train SVM model
    # NOTE: For large datasets, consider using kernel='linear' or sampling the data
    svm_model = train_svm(
        X_train, 
        y_train,
        kernel='rbf',                    # Kernel type: 'linear', 'rbf', 'poly', 'sigmoid'
        C=1.0,                          # Regularization parameter
        gamma='scale',                   # Kernel coefficient
        class_weight='balanced',         # Handle class imbalance
        use_calibration=True,            # Use probability calibration
        random_state=42,
        cache_size=2000                  # MB of cache for kernel computation
    )
    
    # Evaluate model
    evaluation_results = evaluate_model(svm_model, X_train, X_test, y_train, y_test)
    
    # Plot support vectors summary
    plot_support_vectors_summary(svm_model)
    
    # Plot confusion matrix for test set
    plot_confusion_matrix(
        evaluation_results['test_cm'],
        'Confusion Matrix - Test Set (SVM)',
        save_path=os.path.join(RESULTS_DIR, 'confusion_matrix_test_svm.png')
    )
    
    # Plot ROC curve for test set
    plot_roc_curve(
        y_test,
        evaluation_results['y_test_proba'],
        'ROC Curve - Test Set (SVM)',
        save_path=os.path.join(RESULTS_DIR, 'roc_curve_test_svm.png')
    )
    
    # Save model
    save_model(svm_model, MODEL_SAVE_PATH)
    
    # Save results summary
    save_results_summary(
        svm_model,
        evaluation_results,
        os.path.join(RESULTS_DIR, 'results_summary_svm.txt')
    )
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Results saved to: {RESULTS_DIR}/")
    
    return svm_model, evaluation_results


if __name__ == "__main__":
    model, results = main()

