import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
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


def prepare_features_target(df_train, df_test, target_column='HadHeartAttack', validation_split=0.5):
    """
    Separate features and target variable, and create validation set.
    
    IMPORTANT: Since training data is balanced with SMOTE but test data is imbalanced,
    we split the TEST set into validation + test to monitor performance on real imbalanced data.
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Training data (SMOTE-balanced)
    df_test : pd.DataFrame
        Test data (imbalanced, real distribution)
    target_column : str
        Name of the target column
    validation_split : float
        Proportion of TEST data to use for validation (default 0.5 = 50/50 split)
    
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test, class_weight
    """
    # Use full training set (already SMOTE-balanced)
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    
    # Separate features and target for test set
    X_test_full = df_test.drop(columns=[target_column])
    y_test_full = df_test[target_column]
    
    # Split TEST data into validation and test (both keep real imbalanced distribution)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test_full, 
        y_test_full, 
        test_size=(1 - validation_split),
        random_state=42,
        stratify=y_test_full  # Maintain class distribution
    )
    
    print(f"\nFeature columns: {X_train.shape[1]}")
    print(f"Training samples (SMOTE-balanced): {X_train.shape[0]}")
    print(f"Validation samples (real imbalanced): {X_val.shape[0]}")
    print(f"Test samples (real imbalanced): {X_test.shape[0]}")
    print(f"\nTraining class distribution (SMOTE-balanced):\n{y_train.value_counts()}")
    print(f"Training class proportions:\n{y_train.value_counts(normalize=True)}")
    print(f"\nValidation class distribution (real):\n{y_val.value_counts()}")
    print(f"Validation class proportions:\n{y_val.value_counts(normalize=True)}")
    print(f"\nTest class distribution (real):\n{y_test.value_counts()}")
    print(f"Test class proportions:\n{y_test.value_counts(normalize=True)}")
    
    # Calculate class weights based on training data for handling imbalanced data
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    total_count = len(y_train)
    
    class_weight = {
        0: total_count / (2 * neg_count),
        1: total_count / (2 * pos_count)
    }
    
    print(f"\nTraining set class imbalance ratio (neg/pos): {neg_count/pos_count:.2f}")
    print(f"Class weights for training: {class_weight}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_weight


def build_dnn_model(input_dim, architecture='deep', learning_rate=0.001):
    """
    Build a Deep Neural Network model.
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    architecture : str
        Model architecture: 'shallow', 'medium', or 'deep'
    learning_rate : float
        Learning rate for the optimizer
    
    Returns:
    --------
    keras.Model : Compiled DNN model
    """
    print("\n" + "="*60)
    print("Building Deep Neural Network...")
    print("="*60)
    print(f"Architecture: {architecture}")
    print(f"Input dimension: {input_dim}")
    print(f"Learning rate: {learning_rate}")
    
    model = keras.Sequential(name='HeartAttack_DNN')
    
    if architecture == 'shallow':
        # Shallow network - 2 hidden layers
        model.add(layers.Input(shape=(input_dim,), name='input_layer'))
        model.add(layers.Dense(128, activation='relu', 
                              kernel_regularizer=regularizers.l2(0.001),
                              name='hidden_1'))
        model.add(layers.Dropout(0.3, name='dropout_1'))
        model.add(layers.Dense(64, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001),
                              name='hidden_2'))
        model.add(layers.Dropout(0.3, name='dropout_2'))
        
    elif architecture == 'medium':
        # Medium network - 3 hidden layers
        model.add(layers.Input(shape=(input_dim,), name='input_layer'))
        model.add(layers.Dense(256, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001),
                              name='hidden_1'))
        model.add(layers.BatchNormalization(name='batch_norm_1'))
        model.add(layers.Dropout(0.4, name='dropout_1'))
        model.add(layers.Dense(128, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001),
                              name='hidden_2'))
        model.add(layers.BatchNormalization(name='batch_norm_2'))
        model.add(layers.Dropout(0.4, name='dropout_2'))
        model.add(layers.Dense(64, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001),
                              name='hidden_3'))
        model.add(layers.Dropout(0.3, name='dropout_3'))
        
    else:  # deep
        # Deep network - 5 hidden layers
        model.add(layers.Input(shape=(input_dim,), name='input_layer'))
        model.add(layers.Dense(512, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001),
                              name='hidden_1'))
        model.add(layers.BatchNormalization(name='batch_norm_1'))
        model.add(layers.Dropout(0.4, name='dropout_1'))
        model.add(layers.Dense(256, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001),
                              name='hidden_2'))
        model.add(layers.BatchNormalization(name='batch_norm_2'))
        model.add(layers.Dropout(0.4, name='dropout_2'))
        model.add(layers.Dense(128, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001),
                              name='hidden_3'))
        model.add(layers.BatchNormalization(name='batch_norm_3'))
        model.add(layers.Dropout(0.3, name='dropout_3'))
        model.add(layers.Dense(64, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001),
                              name='hidden_4'))
        model.add(layers.Dropout(0.3, name='dropout_4'))
        model.add(layers.Dense(32, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001),
                              name='hidden_5'))
        model.add(layers.Dropout(0.2, name='dropout_5'))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid', name='output_layer'))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    print("\nModel Summary:")
    model.summary()
    
    return model


def train_dnn(model, X_train, y_train, X_val, y_val, class_weight, 
              epochs=100, batch_size=256, patience=15, model_save_path=None):
    """
    Train the Deep Neural Network model.
    
    Parameters:
    -----------
    model : keras.Model
        The DNN model to train
    X_train, y_train : Training data
    X_val, y_val : Validation data
    class_weight : dict
        Class weights for handling imbalance
    epochs : int
        Maximum number of epochs
    batch_size : int
        Batch size for training
    patience : int
        Early stopping patience
    model_save_path : str
        Path to save the best model
    
    Returns:
    --------
    history : keras.callbacks.History
        Training history
    """
    print("\n" + "="*60)
    print("Training Deep Neural Network...")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Early stopping patience: {patience}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    if model_save_path:
        callbacks.append(
            ModelCheckpoint(
                model_save_path.replace('.pkl', '_checkpoint.h5'),
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        )
    
    # Train model
    start_time = datetime.now()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Total epochs trained: {len(history.history['loss'])}")
    
    return history


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Evaluate the model on training, validation, and test sets.
    
    Returns:
    --------
    dict : Dictionary containing all evaluation metrics
    """
    print("\n" + "="*60)
    print("Evaluating Model Performance...")
    print("="*60)
    
    # Make predictions
    y_train_proba = model.predict(X_train, verbose=0).flatten()
    y_val_proba = model.predict(X_val, verbose=0).flatten()
    y_test_proba = model.predict(X_test, verbose=0).flatten()
    
    y_train_pred = (y_train_proba >= 0.5).astype(int)
    y_val_pred = (y_val_proba >= 0.5).astype(int)
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    
    # Calculate metrics for training set
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1': f1_score(y_train, y_train_pred),
        'roc_auc': roc_auc_score(y_train, y_train_proba)
    }
    
    # Calculate metrics for validation set
    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_proba)
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
    print("VALIDATION SET METRICS:")
    print("-"*60)
    for metric, value in val_metrics.items():
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
    val_cm = confusion_matrix(y_val, y_val_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    print("\n" + "-"*60)
    print("CONFUSION MATRIX (Test Set):")
    print("-"*60)
    print(f"True Negatives:  {test_cm[0, 0]:6d}  |  False Positives: {test_cm[0, 1]:6d}")
    print(f"False Negatives: {test_cm[1, 0]:6d}  |  True Positives:  {test_cm[1, 1]:6d}")
    
    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'train_cm': train_cm,
        'val_cm': val_cm,
        'test_cm': test_cm,
        'y_train_pred': y_train_pred,
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
        'y_train_proba': y_train_proba,
        'y_val_proba': y_val_proba,
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


def plot_training_history(history, save_path=None):
    """Plot training and validation metrics over epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot AUC
    axes[1, 0].plot(history.history['auc'], label='Training AUC')
    axes[1, 0].plot(history.history['val_auc'], label='Validation AUC')
    axes[1, 0].set_title('Model AUC', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot Precision and Recall
    axes[1, 1].plot(history.history['precision'], label='Training Precision', linestyle='--')
    axes[1, 1].plot(history.history['val_precision'], label='Validation Precision', linestyle='--')
    axes[1, 1].plot(history.history['recall'], label='Training Recall', linestyle='-.')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linestyle='-.')
    axes[1, 1].set_title('Model Precision & Recall', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    
    plt.show()


def save_model(model, save_path):
    """Save the trained model to disk."""
    # Save as Keras native format
    keras_path = save_path.replace('.pkl', '.keras')
    model.save(keras_path)
    print(f"\nModel saved to {keras_path}")
    
    # Also save as pickle for consistency with other models
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model also saved to {save_path}")


def save_results_summary(model, evaluation_results, history, save_path):
    """Save a summary of results to a text file."""
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DEEP NEURAL NETWORK - HEART ATTACK PREDICTION\n")
        f.write("="*60 + "\n")
        f.write(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("MODEL ARCHITECTURE:\n")
        f.write("-"*60 + "\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        f.write("\n" + "-"*60 + "\n")
        f.write("TRAINING HISTORY:\n")
        f.write("-"*60 + "\n")
        f.write(f"Total epochs trained: {len(history.history['loss'])}\n")
        f.write(f"Final training loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final validation loss: {history.history['val_loss'][-1]:.4f}\n")
        f.write(f"Best validation AUC: {max(history.history['val_auc']):.4f}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("TRAINING SET METRICS:\n")
        f.write("-"*60 + "\n")
        for metric, value in evaluation_results['train_metrics'].items():
            f.write(f"{metric.upper():15s}: {value:.4f}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("VALIDATION SET METRICS:\n")
        f.write("-"*60 + "\n")
        for metric, value in evaluation_results['val_metrics'].items():
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
    """
    Main execution function.
    
    Training Strategy:
    ------------------
    - Training data: SMOTE-balanced (synthetic oversampling)
    - Validation data: Split from test set (real imbalanced distribution)
    - Test data: Split from test set (real imbalanced distribution)
    
    This ensures the model trains on balanced data but is monitored and evaluated
    on real-world imbalanced data during validation and testing.
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Define paths
    TRAIN_PATH = '../resources/heart_2022_train_processed.csv'
    TEST_PATH = '../resources/heart_2022_test_processed.csv'
    MODEL_SAVE_PATH = '../models/dnn_model.pkl'
    RESULTS_DIR = '../results/dnn'
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # Load data
    df_train, df_test = load_data(TRAIN_PATH, TEST_PATH)
    
    # Prepare features and target with validation split
    # NOTE: validation_split applies to TEST set (not training set)
    # Since test is ~20% of total data, 0.5 split gives ~10% val, ~10% test
    X_train, X_val, X_test, y_train, y_val, y_test, class_weight = prepare_features_target(
        df_train, df_test, validation_split=0.5
    )
    
    # Build DNN model
    input_dim = X_train.shape[1]
    dnn_model = build_dnn_model(
        input_dim=input_dim,
        architecture='shallow',  # Options: 'shallow', 'medium', 'deep'
        learning_rate=0.001
    )
    
    # Train DNN model
    history = train_dnn(
        dnn_model,
        X_train, y_train,
        X_val, y_val,
        class_weight=class_weight,
        epochs=100,
        batch_size=256,
        patience=15,
        model_save_path=MODEL_SAVE_PATH
    )
    
    # Plot training history
    plot_training_history(
        history,
        save_path=os.path.join(RESULTS_DIR, 'training_history_dnn.png')
    )
    
    # Evaluate model
    evaluation_results = evaluate_model(
        dnn_model, X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Plot confusion matrix for test set
    plot_confusion_matrix(
        evaluation_results['test_cm'],
        'Confusion Matrix - Test Set (DNN)',
        save_path=os.path.join(RESULTS_DIR, 'confusion_matrix_test_dnn.png')
    )
    
    # Plot confusion matrix for validation set
    plot_confusion_matrix(
        evaluation_results['val_cm'],
        'Confusion Matrix - Validation Set (DNN)',
        save_path=os.path.join(RESULTS_DIR, 'confusion_matrix_val_dnn.png')
    )
    
    # Plot ROC curve for test set
    plot_roc_curve(
        y_test,
        evaluation_results['y_test_proba'],
        'ROC Curve - Test Set (DNN)',
        save_path=os.path.join(RESULTS_DIR, 'roc_curve_test_dnn.png')
    )
    
    # Plot ROC curve for validation set
    plot_roc_curve(
        y_val,
        evaluation_results['y_val_proba'],
        'ROC Curve - Validation Set (DNN)',
        save_path=os.path.join(RESULTS_DIR, 'roc_curve_val_dnn.png')
    )
    
    # Save model
    save_model(dnn_model, MODEL_SAVE_PATH)
    
    # Save results summary
    save_results_summary(
        dnn_model,
        evaluation_results,
        history,
        os.path.join(RESULTS_DIR, 'results_summary_dnn.txt')
    )
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Results saved to: {RESULTS_DIR}/")
    
    return dnn_model, evaluation_results, history


if __name__ == "__main__":
    model, results, history = main()

