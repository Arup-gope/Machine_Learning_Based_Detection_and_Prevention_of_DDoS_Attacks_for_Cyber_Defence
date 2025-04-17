# project.py - DDoS Detection with ML

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import RobustScaler
from itertools import cycle


def load_data():
    """Load dataset from CSV file."""
    df = pd.read_csv("Data/Wednesday-workingHours.pcap_ISCX.csv")
    df.columns = df.columns.str.strip()
    return df


def clean_data(df):
    """Handle missing and infinite values."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df.dropna().copy()
    return df_clean


def preprocess_data(df):
    """Prepare data for machine learning."""
    # Convert labels to numerical values (binary classification)
    label_map = {'BENIGN': 0, 'DoS Hulk': 1, 'DoS GoldenEye': 1,
                 'DoS slowloris': 1, 'DoS Slowhttptest': 1, 'Heartbleed': 1}

    df['Label'] = df['Label'].map(label_map)

    # Remove rows with invalid labels
    df = df.dropna(subset=['Label'])

    # Convert features to numeric and handle missing values
    X = df.drop('Label', axis=1).apply(pd.to_numeric, errors='coerce')
    y = df['Label'].astype(int)

    # Remove columns with all NaN values
    X = X.dropna(axis=1, how='all')

    # Fill remaining missing values with column means
    X = X.fillna(X.mean())

    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def plot_data_distributions(df, title):
    """Plot label distributions."""
    plt.figure(figsize=(8, 4))
    sns.countplot(x='Label', data=df)
    plt.title(f'{title} - Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Add percentage annotations
    total = len(df)
    ax = plt.gca()
    for p in ax.patches:
        percentage = f'{100 * p.get_height()/total:.1f}%'
        ax.annotate(percentage, 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', 
                   xytext=(0, 5), 
                   textcoords='offset points')  # Fixed line

    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{title.replace(" ", "_")}_Distribution.png')
    plt.close()


def save_confusion_matrix(y_true, y_pred, model_name):
    """Save confusion matrix as image."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'DDoS'],
                yticklabels=['Benign', 'DDoS'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name.replace(" ", "_")}_CM.png')
    plt.close()


def save_roc_curves(models, X_test, y_test):
    """Generate and save ROC curve comparison."""
    plt.figure(figsize=(10, 6))
    colors = cycle(['blue', 'green', 'red'])

    for (name, model), color in zip(models.items(), colors):
        try:
            proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color,
                     label=f'{name} (AUC = {roc_auc:.2f})')
        except AttributeError:
            print(f"{name} model does not have predict_proba.")
            continue

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)

    os.makedirs('results', exist_ok=True)
    plt.savefig('results/ROC_Comparison.png')
    plt.close()

# ... (Keep all imports and helper functions above)

if __name__ == "__main__":
    # Data Preparation (Common for All Models)
    # ========================================
    os.makedirs('results', exist_ok=True)
    
    # Load and clean data
    df = load_data()
    df_clean = clean_data(df)
    plot_data_distributions(df_clean, "Raw Data")

    # Preprocess and split data
    X, y = preprocess_data(df_clean)
    plot_data_distributions(pd.DataFrame(X, columns=df_clean.drop('Label', axis=1).columns).assign(Label=y),
                           "Processed Data")

    # Split dataset (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    
    print(f"\n{' Data Summary ':-^40}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Class distribution:\n{pd.Series(y_train).value_counts()}")

    # Random Forest Implementation
    # ============================
    print("\n\n{' Random Forest ':-^40}")
    
    # Initialize and train
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    rf_pred = rf_model.predict(X_test)
    rf_metrics = {
        'Accuracy': accuracy_score(y_test, rf_pred),
        'F1 Score': f1_score(y_test, rf_pred),
        'Precision': precision_score(y_test, rf_pred),
        'Recall': recall_score(y_test, rf_pred)
    }
    
    # Save results
    print("\nEvaluation Metrics:")
    for metric, value in rf_metrics.items():
        print(f"{metric}: {value:.4f}")
        
    save_confusion_matrix(y_test, rf_pred, "Random Forest")

    # Logistic Regression Implementation
    # ==================================
    print("\n\n{' Logistic Regression ':-^40}")
    
    # Initialize and train
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    
    # Evaluate
    lr_pred = lr_model.predict(X_test)
    lr_metrics = {
        'Accuracy': accuracy_score(y_test, lr_pred),
        'F1 Score': f1_score(y_test, lr_pred),
        'Precision': precision_score(y_test, lr_pred),
        'Recall': recall_score(y_test, lr_pred)
    }
    
    # Save results
    print("\nEvaluation Metrics:")
    for metric, value in lr_metrics.items():
        print(f"{metric}: {value:.4f}")
        
    save_confusion_matrix(y_test, lr_pred, "Logistic Regression")

    # Neural Network Implementation
    # =============================
    print("\n\n{' Neural Network ':-^40}")
    
    # Initialize and train
    nn_model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        early_stopping=True,
        random_state=42,
        verbose=True
    )
    nn_model.fit(X_train, y_train)
    
    # Evaluate
    nn_pred = nn_model.predict(X_test)
    nn_metrics = {
        'Accuracy': accuracy_score(y_test, nn_pred),
        'F1 Score': f1_score(y_test, nn_pred),
        'Precision': precision_score(y_test, nn_pred),
        'Recall': recall_score(y_test, nn_pred)
    }
    
    # Save results
    print("\nEvaluation Metrics:")
    for metric, value in nn_metrics.items():
        print(f"{metric}: {value:.4f}")
        
    save_confusion_matrix(y_test, nn_pred, "Neural Network")

    # Comparative Analysis
    # ====================
    print("\n\n{' Final Comparison ':-^40}")
    
    # Collect models for ROC curves
    models = {
        'Random Forest': rf_model,
        'Logistic Regression': lr_model,
        'Neural Network': nn_model
    }
    
    # Generate ROC comparison
    save_roc_curves(models, X_test, y_test)
    
    # Create metrics table
    metrics_data = [
        {'Model': 'Random Forest', **rf_metrics},
        {'Model': 'Logistic Regression', **lr_metrics},
        {'Model': 'Neural Network', **nn_metrics}
    ]
    
    metrics_df = pd.DataFrame(metrics_data).set_index('Model')
    print("\nPerformance Comparison:")
    print(metrics_df.round(4))