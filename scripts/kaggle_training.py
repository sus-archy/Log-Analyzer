#!/usr/bin/env python3
"""
Kaggle Training Script for LogMind AI

This script exports data and creates a Kaggle-compatible training notebook.
Run this locally to prepare data, then upload to Kaggle for fast training.

Usage:
    1. Run: python scripts/kaggle_training.py --export
    2. Upload data/kaggle_export/ to Kaggle as a dataset
    3. Create a new Kaggle notebook and paste the generated code
    4. Run on Kaggle with GPU
    5. Download the trained models
"""

import asyncio
import sys
import json
import csv
import pickle
from pathlib import Path

# Add the api app to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "api"))

EXPORT_DIR = Path("data/kaggle_export")
MAX_EXPORT_LOGS = 500000  # 500k logs is enough for good training


async def export_data_for_kaggle():
    """Export log data to CSV for Kaggle upload."""
    print("=" * 70)
    print("📦 Exporting Data for Kaggle Training")
    print("=" * 70)
    
    from app.storage.db import init_db, close_db, get_db
    
    await init_db()
    db = await get_db()
    
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get total count
    cursor = await db.execute("SELECT COUNT(*) FROM logs_stream")
    row = await cursor.fetchone()
    total = row[0] if row else 0
    print(f"Total logs in database: {total:,}")
    
    # Export logs to CSV
    export_count = min(total, MAX_EXPORT_LOGS)
    print(f"Exporting {export_count:,} logs to CSV...")
    
    csv_path = EXPORT_DIR / "logs_training.csv"
    
    cursor = await db.execute(
        """
        SELECT 
            id, service_name, severity, body_raw, template_hash
        FROM logs_stream 
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (export_count,)
    )
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'service_name', 'severity', 'message', 'template_hash'])
        
        batch_size = 10000
        count = 0
        while True:
            rows = await cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                writer.writerow(row)
                count += 1
            print(f"  Exported {count:,} logs...", end='\r')
    
    print(f"\n✅ Exported {count:,} logs to {csv_path}")
    
    # Get file size
    size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"   File size: {size_mb:.1f} MB")
    
    await close_db()
    
    # Generate Kaggle notebook code
    generate_kaggle_notebook()
    
    print("\n" + "=" * 70)
    print("📋 NEXT STEPS:")
    print("=" * 70)
    print(f"1. Upload '{EXPORT_DIR}/' folder to Kaggle as a dataset")
    print(f"2. Create new Kaggle notebook")
    print(f"3. Copy code from '{EXPORT_DIR}/kaggle_notebook.py'")
    print(f"4. Run with GPU accelerator enabled")
    print(f"5. Download trained models from output")
    print("=" * 70)


def generate_kaggle_notebook():
    """Generate the Python code for Kaggle notebook."""
    
    notebook_code = '''"""
LogMind AI - Kaggle Training Notebook
=====================================
Run this on Kaggle with GPU for fast training!

Features:
- Overfitting detection with train/validation curves
- Regularization (L1/L2) to prevent overfitting
- AUC-ROC curve plotting for model evaluation
- Cross-validation for robust metrics

Dataset: Upload logs_training.csv from your local export
Runtime: GPU P100 or T4 recommended
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🚀 LogMind AI - Kaggle Training")
print("=" * 50)

# ============================================================
# Load Data
# ============================================================
print("\\n📂 Loading data...")

# Adjust path based on your Kaggle dataset name
DATA_PATH = "/kaggle/input/logmind-training-data/logs_training.csv"

try:
    df = pd.read_csv(DATA_PATH)
except:
    # Try alternative path
    DATA_PATH = "/kaggle/input/logs_training.csv"
    df = pd.read_csv(DATA_PATH)

print(f"Loaded {len(df):,} log entries")
print(f"Columns: {list(df.columns)}")
print(f"\\nService distribution:")
print(df['service_name'].value_counts().head(10))

# ============================================================
# Preprocessing
# ============================================================
print("\\n🔧 Preprocessing...")

# Fill missing values
df['message'] = df['message'].fillna('')
df['severity'] = df['severity'].fillna('INFO')
df['service_name'] = df['service_name'].fillna('unknown')

# Map severity to numeric
severity_map = {
    'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'WARN': 2, 
    'ERROR': 3, 'CRITICAL': 4, 'FATAL': 4
}
df['severity_level'] = df['severity'].str.upper().map(severity_map).fillna(1)

# Message length feature
df['message_length'] = df['message'].str.len()

# Template hash as integer (handle overflow)
df['template_id'] = df['template_hash'].apply(
    lambda x: hash(str(x)) % 10000 if pd.notna(x) else 0
)

print(f"Preprocessed {len(df):,} logs")

# ============================================================
# TF-IDF Vectorization
# ============================================================
print("\\n📊 Building TF-IDF features...")

from sklearn.feature_extraction.text import TfidfVectorizer

# Sample for TF-IDF if too large
tfidf_sample = df['message'].sample(min(100000, len(df)), random_state=42)

vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.95,
    ngram_range=(1, 2),
    stop_words='english'
)

tfidf_matrix = vectorizer.fit_transform(tfidf_sample)
print(f"TF-IDF shape: {tfidf_matrix.shape}")

# ============================================================
# Overfitting Detection Setup
# ============================================================
print("\\n📈 Setting up overfitting detection...")

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import os

# Create output directory for plots
os.makedirs('/kaggle/working/plots', exist_ok=True)

def plot_learning_curve(estimator, X, y, title, filename, cv=5, scoring='accuracy'):
    """Plot learning curve to detect overfitting."""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring=scoring
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation Score')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    
    # Detect overfitting
    gap = train_mean[-1] - val_mean[-1]
    if gap > 0.1:
        plt.annotate(f'⚠️ Overfitting detected!\\nGap: {gap:.2%}', 
                     xy=(train_sizes[-1], val_mean[-1]), fontsize=10, color='red')
    
    plt.savefig(f'/kaggle/working/plots/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    
    return train_mean[-1], val_mean[-1], gap

def plot_roc_curve(y_true, y_proba, title, filename, labels=None):
    """Plot ROC curve with AUC score."""
    plt.figure(figsize=(10, 8))
    
    if len(np.unique(y_true)) == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
    else:
        # Multi-class classification - One vs Rest
        from sklearn.preprocessing import label_binarize
        classes = np.unique(y_true)
        y_bin = label_binarize(y_true, classes=classes)
        
        # Handle case where y_proba is 1D
        if y_proba.ndim == 1:
            y_proba = np.column_stack([1 - y_proba, y_proba])
        
        for i, cls in enumerate(classes):
            if i < y_proba.shape[1]:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
                label = labels[i] if labels else f'Class {cls}'
                plt.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'/kaggle/working/plots/{filename}', dpi=150, bbox_inches='tight')
    plt.close()

def plot_train_val_comparison(train_scores, val_scores, model_names, filename):
    """Bar chart comparing training vs validation scores."""
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_scores, width, label='Training', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, val_scores, width, label='Validation', color='orange', alpha=0.7)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Training vs Validation Accuracy (Overfitting Check)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2%}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2%}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    # Highlight overfitting
    for i, (t, v) in enumerate(zip(train_scores, val_scores)):
        if t - v > 0.1:
            ax.annotate('⚠️', xy=(i, max(t, v) + 0.05), ha='center', fontsize=14, color='red')
    
    plt.tight_layout()
    plt.savefig(f'/kaggle/working/plots/{filename}', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# Train Isolation Forest (Anomaly Detection)
# ============================================================
print("\\n🌲 Training Isolation Forest...")

from sklearn.ensemble import IsolationForest

# Prepare features for anomaly detection
anomaly_features = df[['severity_level', 'message_length', 'template_id']].values

# Train Isolation Forest with regularization via contamination tuning
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    max_samples='auto',  # Regularization: subsample for each tree
    max_features=1.0,    # Use all features
    bootstrap=False,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
iso_forest.fit(anomaly_features)
print("✅ Isolation Forest trained")

# ============================================================
# Train Naive Bayes Classifiers with Overfitting Detection
# ============================================================
print("\\n📝 Training Naive Bayes classifiers with regularization...")

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Store metrics for comparison
train_accuracies = []
val_accuracies = []
model_names = []

# Severity classifier with regularization (alpha smoothing)
severity_labels = tfidf_sample.index.map(lambda i: df.loc[i, 'severity'])
X_train, X_val, y_train, y_val = train_test_split(
    tfidf_matrix, severity_labels,
    test_size=0.2, random_state=42, stratify=severity_labels
)

# Try different alpha values for regularization
alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
best_alpha = 1.0
best_val_score = 0

print("  Tuning regularization (alpha) for Severity classifier...")
for alpha in alphas:
    clf = MultinomialNB(alpha=alpha)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    if cv_scores.mean() > best_val_score:
        best_val_score = cv_scores.mean()
        best_alpha = alpha
    print(f"    alpha={alpha}: CV accuracy = {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

print(f"  Best alpha: {best_alpha}")

severity_classifier = MultinomialNB(alpha=best_alpha)
severity_classifier.fit(X_train, y_train)

severity_train_acc = severity_classifier.score(X_train, y_train)
severity_val_acc = severity_classifier.score(X_val, y_val)
print(f"✅ Severity classifier - Train: {severity_train_acc:.2%}, Val: {severity_val_acc:.2%}")

train_accuracies.append(severity_train_acc)
val_accuracies.append(severity_val_acc)
model_names.append('Severity\\nClassifier')

# Plot learning curve for severity classifier
print("  Plotting learning curve for Severity classifier...")
plot_learning_curve(severity_classifier, X_train, y_train, 
                    'Severity Classifier Learning Curve', 'severity_learning_curve.png')

# Get predictions for ROC curve
severity_proba = severity_classifier.predict_proba(X_val)
plot_roc_curve(y_val, severity_proba, 'Severity Classifier ROC Curves', 
               'severity_roc_curve.png', labels=severity_classifier.classes_)

# Domain classifier with regularization
domain_labels = tfidf_sample.index.map(lambda i: df.loc[i, 'service_name'])
X_train, X_val, y_train, y_val = train_test_split(
    tfidf_matrix, domain_labels, test_size=0.2, random_state=42
)

# Use Logistic Regression with L2 regularization for better control
print("\\n  Training Domain classifier with L2 regularization...")

# Find optimal C (inverse of regularization strength)
from sklearn.model_selection import GridSearchCV

domain_classifier_lr = LogisticRegression(
    penalty='l2',
    C=1.0,  # Default, will tune
    solver='lbfgs',
    max_iter=500,
    multi_class='multinomial',
    random_state=42,
    n_jobs=-1
)

# Quick grid search for regularization
param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}
grid_search = GridSearchCV(domain_classifier_lr, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"  Best C for Domain classifier: {grid_search.best_params_['C']}")

domain_classifier = grid_search.best_estimator_
domain_train_acc = domain_classifier.score(X_train, y_train)
domain_val_acc = domain_classifier.score(X_val, y_val)
print(f"✅ Domain classifier - Train: {domain_train_acc:.2%}, Val: {domain_val_acc:.2%}")

train_accuracies.append(domain_train_acc)
val_accuracies.append(domain_val_acc)
model_names.append('Domain\\nClassifier')

# Plot learning curve for domain classifier  
print("  Plotting learning curve for Domain classifier...")
plot_learning_curve(domain_classifier, X_train, y_train,
                    'Domain Classifier Learning Curve', 'domain_learning_curve.png')

# ============================================================
# Train Security Threat Detector with Regularization
# ============================================================
print("\\n🔒 Training Security Threat Detector with regularization...")

# Security keywords for labeling
security_keywords = [
    'attack', 'malicious', 'unauthorized', 'breach', 'exploit',
    'injection', 'xss', 'csrf', 'sql injection', 'brute force',
    'ddos', 'dos', 'flood', 'intrusion', 'vulnerability',
    'failed login', 'authentication failed', 'access denied',
    'permission denied', 'invalid password', 'root', 'sudo',
    'shell', 'exec', 'eval', 'script', '../', 'etc/passwd'
]

def has_security_keywords(msg):
    msg_lower = str(msg).lower()
    return any(kw in msg_lower for kw in security_keywords)

df['is_security_threat'] = df['message'].apply(has_security_keywords)
security_ratio = df['is_security_threat'].mean()
print(f"Security threats in data: {security_ratio:.2%}")

# Train on TF-IDF features
threat_labels = tfidf_sample.index.map(lambda i: df.loc[i, 'is_security_threat'])

from sklearn.linear_model import SGDClassifier

# SGD with L2 regularization and early stopping for overfitting prevention
X_train, X_val, y_train, y_val = train_test_split(
    tfidf_matrix, threat_labels, test_size=0.2, random_state=42, stratify=threat_labels
)

# Tune regularization
print("  Tuning regularization for Security detector...")
alpha_values = [0.00001, 0.0001, 0.001, 0.01]
best_alpha_security = 0.0001
best_val_auc = 0

for alpha in alpha_values:
    clf = SGDClassifier(
        loss='log_loss',
        penalty='elasticnet',  # Mix of L1 and L2
        l1_ratio=0.15,         # 15% L1, 85% L2
        alpha=alpha,
        early_stopping=True,   # Prevent overfitting
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    val_proba = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    print(f"    alpha={alpha}: Val AUC = {val_auc:.4f}")
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_alpha_security = alpha

print(f"  Best alpha: {best_alpha_security}")

security_classifier = SGDClassifier(
    loss='log_loss',
    penalty='elasticnet',
    l1_ratio=0.15,
    alpha=best_alpha_security,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5,
    random_state=42,
    n_jobs=-1
)
security_classifier.fit(X_train, y_train)

security_train_acc = security_classifier.score(X_train, y_train)
security_val_acc = security_classifier.score(X_val, y_val)
print(f"✅ Security classifier - Train: {security_train_acc:.2%}, Val: {security_val_acc:.2%}")

train_accuracies.append(security_train_acc)
val_accuracies.append(security_val_acc)
model_names.append('Security\\nDetector')

# Plot ROC curve for security classifier
security_proba = security_classifier.predict_proba(X_val)[:, 1]
security_auc = roc_auc_score(y_val, security_proba)
print(f"  Security AUC-ROC: {security_auc:.4f}")

plt.figure(figsize=(10, 8))
fpr, tpr, thresholds = roc_curve(y_val, security_proba)
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Security Detector (AUC = {security_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.fill_between(fpr, 0, tpr, alpha=0.1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Security Threat Detector - ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('/kaggle/working/plots/security_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot learning curve for security classifier
print("  Plotting learning curve for Security detector...")
plot_learning_curve(security_classifier, X_train, y_train,
                    'Security Detector Learning Curve', 'security_learning_curve.png', 
                    scoring='roc_auc')

# ============================================================
# Overfitting Summary Plot
# ============================================================
print("\\n📊 Generating overfitting summary plot...")

plot_train_val_comparison(train_accuracies, val_accuracies, model_names, 
                          'overfitting_summary.png')

# Check for overfitting
print("\\n" + "=" * 50)
print("🔍 OVERFITTING ANALYSIS")
print("=" * 50)
for name, train_acc, val_acc in zip(model_names, train_accuracies, val_accuracies):
    gap = train_acc - val_acc
    status = "⚠️ OVERFITTING" if gap > 0.1 else "✅ OK" if gap > 0.05 else "✅ GOOD FIT"
    print(f"{name.replace(chr(10), ' ')}: Train={train_acc:.2%}, Val={val_acc:.2%}, Gap={gap:.2%} {status}")

# ============================================================
# Save Models
# ============================================================
print("\\n💾 Saving models...")

os.makedirs('/kaggle/working/models', exist_ok=True)

# Save all models
models = {
    'anomaly_detector': {
        'isolation_forest': iso_forest,
        'feature_names': ['severity_level', 'message_length', 'template_id'],
        'threshold': 0.5
    },
    'log_classifier': {
        'vectorizer': vectorizer,
        'severity_classifier': severity_classifier,
        'domain_classifier': domain_classifier,
        'severity_accuracy': severity_val_acc,
        'domain_accuracy': domain_val_acc,
        'severity_train_accuracy': severity_train_acc,
        'domain_train_accuracy': domain_train_acc,
        'regularization': {
            'severity_alpha': best_alpha,
            'domain_C': grid_search.best_params_['C']
        }
    },
    'security_detector': {
        'classifier': security_classifier,
        'vectorizer': vectorizer,
        'keywords': security_keywords,
        'accuracy': security_val_acc,
        'train_accuracy': security_train_acc,
        'auc_score': security_auc,
        'regularization': {
            'alpha': best_alpha_security,
            'penalty': 'elasticnet',
            'l1_ratio': 0.15
        }
    }
}

for name, model in models.items():
    path = f'/kaggle/working/models/{name}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✅ Saved {name}.pkl")

# ============================================================
# Summary
# ============================================================
print("\\n" + "=" * 50)
print("🎉 TRAINING COMPLETE!")
print("=" * 50)
print(f"Logs trained on: {len(df):,}")
print(f"TF-IDF features: {tfidf_matrix.shape[1]:,}")
print(f"\\n📊 Model Performance (Validation):")
print(f"  Severity accuracy: {severity_val_acc:.2%}")
print(f"  Domain accuracy: {domain_val_acc:.2%}")
print(f"  Security accuracy: {security_val_acc:.2%}")
print(f"  Security AUC-ROC: {security_auc:.4f}")
print(f"\\n📈 Overfitting Check (Train - Val gap):")
print(f"  Severity: {severity_train_acc - severity_val_acc:.2%}")
print(f"  Domain: {domain_train_acc - domain_val_acc:.2%}")
print(f"  Security: {security_train_acc - security_val_acc:.2%}")
print("\\n📁 Output files:")
print("  Models: /kaggle/working/models/")
print("  Plots: /kaggle/working/plots/")
print("    - severity_learning_curve.png")
print("    - severity_roc_curve.png")
print("    - domain_learning_curve.png")
print("    - security_learning_curve.png")
print("    - security_roc_curve.png")
print("    - overfitting_summary.png")
print("=" * 50)
'''
    
    notebook_path = EXPORT_DIR / "kaggle_notebook.py"
    notebook_path.write_text(notebook_code)
    print(f"✅ Generated Kaggle notebook: {notebook_path}")
    
    # Also create a simple Jupyter notebook format
    notebook_json = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": notebook_code.split('\n')
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    ipynb_path = EXPORT_DIR / "logmind_training.ipynb"
    with open(ipynb_path, 'w') as f:
        json.dump(notebook_json, f, indent=2)
    print(f"✅ Generated Jupyter notebook: {ipynb_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        asyncio.run(export_data_for_kaggle())
    else:
        print("Usage: python scripts/kaggle_training.py --export")
        print("\nThis will:")
        print("  1. Export log data to CSV")
        print("  2. Generate Kaggle notebook code")
        print("  3. Provide instructions for Kaggle upload")
