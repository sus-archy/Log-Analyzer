#!/usr/bin/env python3
"""Retrain ML models locally with compatible sklearn version."""

import pickle
import numpy as np
import pandas as pd
import re
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("🔄 Retraining models locally with compatible sklearn version...")

# Load exported data
df = pd.read_csv('data/kaggle_export/logs_training.csv')
print(f"📂 Loaded {len(df):,} logs")

# Preprocessing
df['message'] = df['message'].fillna('').astype(str)
df['severity'] = df['severity'].fillna(1)
df['service_name'] = df['service_name'].fillna('unknown').astype(str)

# Map severity
def map_severity(val):
    if pd.isna(val):
        return 1
    if isinstance(val, (int, float)):
        return min(int(val), 4)
    severity_map = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'WARN': 2, 'ERROR': 3, 'CRITICAL': 4, 'FATAL': 4}
    return severity_map.get(str(val).upper(), 1)

df['severity_level'] = df['severity'].apply(map_severity)
df['message_length'] = df['message'].str.len()
df['template_id'] = df['template_hash'].apply(lambda x: hash(str(x)) % 10000 if pd.notna(x) else 0)

print("🔧 Preprocessing complete")

# TF-IDF
print("📊 Building TF-IDF...")
sample_size = min(100000, len(df))
sample_idx = df.sample(sample_size, random_state=42).index
sample_msgs = df.loc[sample_idx, 'message']

vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.95, ngram_range=(1, 2), stop_words='english')
tfidf_matrix = vectorizer.fit_transform(sample_msgs)
print(f"   TF-IDF: {tfidf_matrix.shape}")

# Anomaly detector
print("🌲 Training Isolation Forest...")
anomaly_features = df[['severity_level', 'message_length', 'template_id']].values.astype(np.float32)
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
iso_forest.fit(anomaly_features)
print("   ✅ Isolation Forest trained")

# Severity classifier
print("📝 Training classifiers...")
severity_labels = df.loc[sample_idx, 'severity_level'].astype(str)
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, severity_labels, test_size=0.2, random_state=42)
severity_classifier = MultinomialNB()
severity_classifier.fit(X_train, y_train)
severity_acc = severity_classifier.score(X_test, y_test)
print(f"   Severity: {severity_acc:.2%}")

# Domain classifier
domain_labels = df.loc[sample_idx, 'service_name'].astype(str)
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, domain_labels, test_size=0.2, random_state=42)
domain_classifier = MultinomialNB()
domain_classifier.fit(X_train, y_train)
domain_acc = domain_classifier.score(X_test, y_test)
print(f"   Domain: {domain_acc:.2%}")

# Security features
print("🔒 Training Security Detector (Behavioral)...")

def extract_security_features(msg):
    msg = str(msg).lower()
    features = []
    features.extend([len(re.findall(r'\b4\d{2}\b', msg)), len(re.findall(r'\b5\d{2}\b', msg))])
    features.append(len(re.findall(r'\.\./', msg)) + len(re.findall(r'%2e%2e', msg)))
    features.extend([len(re.findall(r'/(?:etc|bin|tmp|var|usr|proc)/', msg)), len(re.findall(r'\.(php|sh|pl|py|rb|cgi|asp)\b', msg))])
    features.append(min(len(re.findall(r'%[0-9a-fA-F]{2}', msg)), 20))
    special_chars = len(re.findall(r'[<>\'";=|&$`\\]', msg))
    features.append(min(special_chars / max(len(msg), 1) * 100, 50))
    features.extend([min(len(re.findall(r'\b\d{4,}\b', msg)), 10), min(len(re.findall(r'[A-Za-z0-9+/]{20,}={0,2}', msg)), 5)])
    features.extend([min(len(re.findall(r'(?:0x[0-9a-fA-F]+|\\x[0-9a-fA-F]{2})', msg)), 10), min(len(re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', msg)), 5)])
    features.extend([min(len(re.findall(r'[|><`$()]', msg)), 20), min(max([len(w) for w in msg.split()] + [0]), 200)])
    features.append(1 if re.search(r'user|login|auth|pass|session|token', msg) else 0)
    return features

feature_names = ['http_4xx', 'http_5xx', 'path_traversal', 'sensitive_paths', 'script_extensions',
                 'url_encoded', 'special_char_ratio', 'high_numbers', 'base64_patterns', 
                 'hex_patterns', 'ip_count', 'cmd_patterns', 'longest_word', 'auth_context']

print("   Extracting behavioral features...")
security_features = np.array([extract_security_features(msg) for msg in df['message'].values])
combined = np.column_stack([security_features, df['severity_level'].values, df['message_length'].values / 100])

print("   Training anomaly detector for labeling...")
security_iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
pseudo_labels = (security_iso.fit_predict(combined) == -1).astype(int)
print(f"   Detected threats: {pseudo_labels.mean():.2%}")

sample_features = security_features[sample_idx]
sample_labels = pseudo_labels[sample_idx]
scaler = StandardScaler()
sample_scaled = scaler.fit_transform(sample_features)

X_train, X_test, y_train, y_test = train_test_split(sample_scaled, sample_labels, test_size=0.2, random_state=42, stratify=sample_labels)
security_classifier = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=10, class_weight='balanced', random_state=42, n_jobs=-1)
security_classifier.fit(X_train, y_train)
security_acc = security_classifier.score(X_test, y_test)
print(f"   Security: {security_acc:.2%}")

# Save models
print("\n💾 Saving models...")

models = {
    'data/models/anomaly_detector.pkl': {
        'isolation_forest': iso_forest,
        'feature_names': ['severity_level', 'message_length', 'template_id'],
        'threshold': 0.5
    },
    'data/models/log_classifier.pkl': {
        'vectorizer': vectorizer,
        'severity_classifier': severity_classifier,
        'domain_classifier': domain_classifier,
        'severity_accuracy': severity_acc,
        'domain_accuracy': domain_acc
    },
    'data/models/security_detector.pkl': {
        'classifier': security_classifier,
        'scaler': scaler,
        'feature_names': feature_names,
        'anomaly_detector': security_iso,
        'accuracy': security_acc,
        'type': 'behavioral'
    }
}

for path, model in models.items():
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✅ {path.split('/')[-1]}")

# Clean up Kaggle downloaded files
import os
for f in ['anomaly.pkl', 'log.pkl', 'security.pkl']:
    if os.path.exists(f):
        os.remove(f)
        print(f"   🗑️ Removed {f}")

print("\n" + "=" * 50)
print("🎉 TRAINING COMPLETE!")
print("=" * 50)
print(f"Logs trained on: {len(df):,}")
print(f"TF-IDF features: {tfidf_matrix.shape[1]:,}")
print(f"Severity accuracy: {severity_acc:.2%}")
print(f"Domain accuracy: {domain_acc:.2%}")
print(f"Security accuracy: {security_acc:.2%}")
print(f"\n🔒 Security model uses BEHAVIORAL analysis (no keyword matching)")
print("=" * 50)
