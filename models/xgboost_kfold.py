import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
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
    
    # Calculate scale_pos_weight for handling imbalanced data
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"\nClass imbalance ratio (neg/pos): {scale_pos_weight:.2f}")
    print(f"Recommended scale_pos_weight: {scale_pos_weight:.2f}")
    
    return X_train, X_test, y_train, y_test, scale_pos_weight


def train_and_evaluate_kfold(X_train, y_train, n_splits=5, random_state=42, **model_params):
    """
    Train XGBoost using k-fold cross validation.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    n_splits : int
        Number of folds for cross validation
    random_state : int
        Random state for reproducibility
    **model_params : dict
        Parameters for XGBClassifier
    
    Returns:
    --------
    dict : Dictionary containing fold metrics and trained models
    """
    print("\n" + "="*60)
    print(f"K-FOLD CROSS VALIDATION (k={n_splits})")
    print("="*60)
    
    # Initialize StratifiedKFold to maintain class distribution
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Storage for results
    fold_results = {
        'models': [],
        'train_metrics': [],
        'val_metrics': [],
        'confusion_matrices': [],
        'feature_importances': []
    }
    
    # Iterate through folds
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print("\n" + "-"*60)
        print(f"FOLD {fold_idx}/{n_splits}")
        print("-"*60)
        
        # Split data for this fold
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        print(f"Train samples: {len(X_fold_train)}")
        print(f"Validation samples: {len(X_fold_val)}")
        print(f"Train class distribution: {y_fold_train.value_counts().to_dict()}")
        print(f"Val class distribution: {y_fold_val.value_counts().to_dict()}")
        
        # Initialize and train model
        xgb_model = xgb.XGBClassifier(**model_params)
        
        start_time = datetime.now()
        xgb_model.fit(X_fold_train, y_fold_train, verbose=False)
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"Training time: {training_time:.2f} seconds")
        
        # Make predictions
        y_fold_train_pred = xgb_model.predict(X_fold_train)
        y_fold_val_pred = xgb_model.predict(X_fold_val)
        
        y_fold_train_proba = xgb_model.predict_proba(X_fold_train)[:, 1]
        y_fold_val_proba = xgb_model.predict_proba(X_fold_val)[:, 1]
        
        # Calculate metrics for training set
        train_metrics = {
            'accuracy': accuracy_score(y_fold_train, y_fold_train_pred),
            'precision': precision_score(y_fold_train, y_fold_train_pred),
            'recall': recall_score(y_fold_train, y_fold_train_pred),
            'f1': f1_score(y_fold_train, y_fold_train_pred),
            'roc_auc': roc_auc_score(y_fold_train, y_fold_train_proba)
        }
        
        # Calculate metrics for validation set
        val_metrics = {
            'accuracy': accuracy_score(y_fold_val, y_fold_val_pred),
            'precision': precision_score(y_fold_val, y_fold_val_pred),
            'recall': recall_score(y_fold_val, y_fold_val_pred),
            'f1': f1_score(y_fold_val, y_fold_val_pred),
            'roc_auc': roc_auc_score(y_fold_val, y_fold_val_proba)
        }
        
        # Print fold metrics
        print("\nFold Train Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric.upper():15s}: {value:.4f}")
        
        print("\nFold Validation Metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric.upper():15s}: {value:.4f}")
        
        # Get feature importance
        importance_dict = xgb_model.get_booster().get_score(importance_type='weight')
        feature_names = X_train.columns.tolist()
        importances = np.zeros(len(feature_names))
        for i, feature in enumerate(feature_names):
            key = f'f{i}'
            if key in importance_dict:
                importances[i] = importance_dict[key]
        
        # Store results
        fold_results['models'].append(xgb_model)
        fold_results['train_metrics'].append(train_metrics)
        fold_results['val_metrics'].append(val_metrics)
        fold_results['confusion_matrices'].append(confusion_matrix(y_fold_val, y_fold_val_pred))
        fold_results['feature_importances'].append(importances)
    
    return fold_results


def summarize_kfold_results(fold_results, n_splits):
    """
    Calculate and display summary statistics across all folds.
    
    Parameters:
    -----------
    fold_results : dict
        Dictionary containing results from all folds
    n_splits : int
        Number of folds
    """
    print("\n" + "="*60)
    print("K-FOLD CROSS VALIDATION SUMMARY")
    print("="*60)
    
    # Extract metrics from all folds
    metric_names = fold_results['train_metrics'][0].keys()
    
    train_summary = {}
    val_summary = {}
    
    for metric in metric_names:
        train_values = [fold[metric] for fold in fold_results['train_metrics']]
        val_values = [fold[metric] for fold in fold_results['val_metrics']]
        
        train_summary[metric] = {
            'mean': np.mean(train_values),
            'std': np.std(train_values),
            'min': np.min(train_values),
            'max': np.max(train_values)
        }
        
        val_summary[metric] = {
            'mean': np.mean(val_values),
            'std': np.std(val_values),
            'min': np.min(val_values),
            'max': np.max(val_values)
        }
    
    # Print summary
    print("\n" + "-"*60)
    print("TRAINING SET SUMMARY (across all folds):")
    print("-"*60)
    for metric in metric_names:
        stats = train_summary[metric]
        print(f"{metric.upper():15s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
              f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")
    
    print("\n" + "-"*60)
    print("VALIDATION SET SUMMARY (across all folds):")
    print("-"*60)
    for metric in metric_names:
        stats = val_summary[metric]
        print(f"{metric.upper():15s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
              f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")
    
    # Calculate average confusion matrix
    avg_cm = np.mean(fold_results['confusion_matrices'], axis=0).astype(int)
    print("\n" + "-"*60)
    print("AVERAGE CONFUSION MATRIX (Validation Sets):")
    print("-"*60)
    print(f"True Negatives:  {avg_cm[0, 0]:6d}  |  False Positives: {avg_cm[0, 1]:6d}")
    print(f"False Negatives: {avg_cm[1, 0]:6d}  |  True Positives:  {avg_cm[1, 1]:6d}")
    
    return train_summary, val_summary, avg_cm


def evaluate_on_test_set(models, X_test, y_test):
    """
    Evaluate all fold models on the held-out test set using ensemble predictions.
    
    Parameters:
    -----------
    models : list
        List of trained models from each fold
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    """
    print("\n" + "="*60)
    print("EVALUATION ON HELD-OUT TEST SET (Ensemble)")
    print("="*60)
    
    # Get predictions from all models
    all_probas = []
    for model in models:
        y_proba = model.predict_proba(X_test)[:, 1]
        all_probas.append(y_proba)
    
    # Average predictions (ensemble)
    y_test_proba_ensemble = np.mean(all_probas, axis=0)
    y_test_pred_ensemble = (y_test_proba_ensemble >= 0.5).astype(int)
    
    # Calculate metrics
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred_ensemble),
        'precision': precision_score(y_test, y_test_pred_ensemble),
        'recall': recall_score(y_test, y_test_pred_ensemble),
        'f1': f1_score(y_test, y_test_pred_ensemble),
        'roc_auc': roc_auc_score(y_test, y_test_proba_ensemble)
    }
    
    print("\nTest Set Metrics (Ensemble of all folds):")
    for metric, value in test_metrics.items():
        print(f"  {metric.upper():15s}: {value:.4f}")
    
    # Confusion matrix
    test_cm = confusion_matrix(y_test, y_test_pred_ensemble)
    print("\nConfusion Matrix (Test Set):")
    print(f"True Negatives:  {test_cm[0, 0]:6d}  |  False Positives: {test_cm[0, 1]:6d}")
    print(f"False Negatives: {test_cm[1, 0]:6d}  |  True Positives:  {test_cm[1, 1]:6d}")
    
    # Classification report
    print("\nDetailed Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred_ensemble, 
                              target_names=['No Heart Attack', 'Heart Attack']))
    
    return test_metrics, test_cm, y_test_pred_ensemble, y_test_proba_ensemble


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
    
    plt.close()


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
    
    plt.close()


def plot_feature_importance(fold_results, feature_names, top_n=20, save_path=None):
    """Plot average feature importance across all folds."""
    # Calculate average feature importance
    avg_importances = np.mean(fold_results['feature_importances'], axis=0)
    std_importances = np.std(fold_results['feature_importances'], axis=0)
    
    # Normalize
    if avg_importances.sum() > 0:
        avg_importances_norm = avg_importances / avg_importances.sum()
        std_importances_norm = std_importances / avg_importances.sum()
    else:
        avg_importances_norm = avg_importances
        std_importances_norm = std_importances
    
    # Sort by importance
    indices = np.argsort(avg_importances_norm)[::-1]
    
    # Select top N features
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = avg_importances_norm[top_indices]
    top_stds = std_importances_norm[top_indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_importances[::-1], align='center', 
             color='steelblue', xerr=top_stds[::-1], capsize=3)
    plt.yticks(range(top_n), [top_features[i] for i in range(top_n-1, -1, -1)])
    plt.xlabel('Average Normalized Feature Importance (across folds)', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features (XGBoost)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")
    
    plt.close()
    
    # Print feature importances
    print("\n" + "-"*60)
    print(f"TOP {top_n} FEATURE IMPORTANCES (Average across folds):")
    print("-"*60)
    for i, (feature, importance, std) in enumerate(zip(top_features, top_importances, top_stds), 1):
        print(f"{i:2d}. {feature:50s}: {importance:.6f} ± {std:.6f}")


def plot_cv_metrics(fold_results, save_path=None):
    """Plot metrics across folds."""
    metric_names = list(fold_results['val_metrics'][0].keys())
    n_folds = len(fold_results['val_metrics'])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metric_names):
        train_values = [fold[metric] for fold in fold_results['train_metrics']]
        val_values = [fold[metric] for fold in fold_results['val_metrics']]
        
        x = np.arange(1, n_folds + 1)
        axes[idx].plot(x, train_values, marker='o', label='Train', linewidth=2)
        axes[idx].plot(x, val_values, marker='s', label='Validation', linewidth=2)
        axes[idx].set_xlabel('Fold', fontsize=10)
        axes[idx].set_ylabel(metric.upper(), fontsize=10)
        axes[idx].set_title(f'{metric.upper()} across Folds', fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
        axes[idx].set_xticks(x)
    
    # Remove extra subplot
    if len(metric_names) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved CV metrics plot to {save_path}")
    
    plt.close()


def save_results_summary(fold_results, train_summary, val_summary, 
                        test_metrics, n_splits, model_params, save_path):
    """Save comprehensive results summary to a text file."""
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("XGBOOST WITH K-FOLD CROSS VALIDATION\n")
        f.write("HEART ATTACK PREDICTION\n")
        f.write("="*60 + "\n")
        f.write(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Folds: {n_splits}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("MODEL PARAMETERS:\n")
        f.write("-"*60 + "\n")
        for key, value in model_params.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("TRAINING SET SUMMARY (across all folds):\n")
        f.write("-"*60 + "\n")
        for metric, stats in train_summary.items():
            f.write(f"{metric.upper():15s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                   f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("VALIDATION SET SUMMARY (across all folds):\n")
        f.write("-"*60 + "\n")
        for metric, stats in val_summary.items():
            f.write(f"{metric.upper():15s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                   f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("TEST SET METRICS (Ensemble):\n")
        f.write("-"*60 + "\n")
        for metric, value in test_metrics.items():
            f.write(f"{metric.upper():15s}: {value:.4f}\n")
        
        # Write individual fold results
        f.write("\n" + "="*60 + "\n")
        f.write("INDIVIDUAL FOLD RESULTS:\n")
        f.write("="*60 + "\n")
        for i, (train_metrics, val_metrics) in enumerate(zip(fold_results['train_metrics'], 
                                                             fold_results['val_metrics']), 1):
            f.write(f"\nFold {i}:\n")
            f.write("  Training:\n")
            for metric, value in train_metrics.items():
                f.write(f"    {metric.upper():15s}: {value:.4f}\n")
            f.write("  Validation:\n")
            for metric, value in val_metrics.items():
                f.write(f"    {metric.upper():15s}: {value:.4f}\n")
    
    print(f"Results summary saved to {save_path}")


def main():
    """Main execution function."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define paths
    TRAIN_PATH = '../resources/heart_2022_train_processed.csv'
    TEST_PATH = '../resources/heart_2022_test_processed.csv'
    RESULTS_DIR = '../results/xgboost_kfold'
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load data
    df_train, df_test = load_data(TRAIN_PATH, TEST_PATH)
    
    # Prepare features and target
    X_train, X_test, y_train, y_test, scale_pos_weight = prepare_features_target(df_train, df_test)
    
    # Define model parameters
    model_params = {
        'n_estimators': 1100,
        'max_depth': 5,
        'learning_rate': 0.008,
        'random_state': 42,
    }
    
    print("\nModel Parameters:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    
    # Perform k-fold cross validation
    n_splits = 5
    fold_results = train_and_evaluate_kfold(
        X_train, 
        y_train, 
        n_splits=n_splits,
        **model_params
    )
    
    # Summarize cross-validation results
    train_summary, val_summary, avg_cm = summarize_kfold_results(fold_results, n_splits)
    
    # Evaluate ensemble on test set
    test_metrics, test_cm, y_test_pred, y_test_proba = evaluate_on_test_set(
        fold_results['models'], X_test, y_test
    )
    
    # Plot average confusion matrix from CV
    plot_confusion_matrix(
        avg_cm,
        f'Average Confusion Matrix - CV Validation ({n_splits}-Fold)',
        save_path=os.path.join(RESULTS_DIR, 'confusion_matrix_cv_avg.png')
    )
    
    # Plot confusion matrix for test set
    plot_confusion_matrix(
        test_cm,
        'Confusion Matrix - Test Set (Ensemble)',
        save_path=os.path.join(RESULTS_DIR, 'confusion_matrix_test.png')
    )
    
    # Plot ROC curve for test set
    plot_roc_curve(
        y_test,
        y_test_proba,
        'ROC Curve - Test Set (Ensemble)',
        save_path=os.path.join(RESULTS_DIR, 'roc_curve_test.png')
    )
    
    # Plot feature importance (average across folds)
    plot_feature_importance(
        fold_results,
        X_train.columns.tolist(),
        top_n=20,
        save_path=os.path.join(RESULTS_DIR, 'feature_importance_avg.png')
    )
    
    # Plot metrics across folds
    plot_cv_metrics(
        fold_results,
        save_path=os.path.join(RESULTS_DIR, 'metrics_across_folds.png')
    )
    
    # Save results summary
    save_results_summary(
        fold_results,
        train_summary,
        val_summary,
        test_metrics,
        n_splits,
        model_params,
        os.path.join(RESULTS_DIR, 'results_summary_kfold.txt')
    )
    
    # Save the ensemble models
    ensemble_save_path = os.path.join(RESULTS_DIR, 'xgb_ensemble_models.pkl')
    with open(ensemble_save_path, 'wb') as f:
        pickle.dump(fold_results['models'], f)
    print(f"\nEnsemble models saved to {ensemble_save_path}")
    
    print("\n" + "="*60)
    print("K-FOLD CROSS VALIDATION COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"\nKey Findings:")
    print(f"  - Validation Accuracy: {val_summary['accuracy']['mean']:.4f} ± {val_summary['accuracy']['std']:.4f}")
    print(f"  - Validation ROC-AUC:  {val_summary['roc_auc']['mean']:.4f} ± {val_summary['roc_auc']['std']:.4f}")
    print(f"  - Test Set Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"  - Test Set ROC-AUC:    {test_metrics['roc_auc']:.4f}")
    
    return fold_results, test_metrics


if __name__ == "__main__":
    fold_results, test_metrics = main()

