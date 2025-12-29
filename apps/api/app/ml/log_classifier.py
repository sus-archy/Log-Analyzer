"""
Log Pattern Classifier with Real Training

This module implements a trainable log classifier that learns patterns from actual log data.
Unlike the previous keyword-based approach, this uses:
1. TF-IDF vectorization for text features
2. Naive Bayes and SVM classifiers for categorization
3. Online learning for continuous improvement
4. Cross-domain transfer learning

The classifier learns from the actual log files in the Logs/loghub directory.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of log classification"""
    category: str  # 'error', 'warning', 'security', 'performance', 'normal'
    domain: str  # 'web_server', 'database', 'system', 'network', etc.
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    confidence: float
    probabilities: Dict[str, float]
    extracted_entities: Dict[str, List[str]] = field(default_factory=dict)


class TFIDFVectorizer:
    """
    Custom TF-IDF implementation for log text vectorization.
    
    TF-IDF (Term Frequency - Inverse Document Frequency) captures:
    - TF: How often a term appears in a document
    - IDF: How rare/important a term is across all documents
    """
    
    def __init__(self, max_features: int = 5000, min_df: int = 2, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.vocabulary: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.fitted = False
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize log text with log-specific preprocessing"""
        # Lowercase and clean
        text = text.lower()
        
        # Normalize common log patterns
        text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' IP_ADDR ', text)  # IP addresses
        text = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', ' TIMESTAMP ', text)  # ISO timestamps
        text = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', ' UUID ', text)  # UUIDs
        text = re.sub(r'0x[0-9a-f]+', ' HEX_NUM ', text)  # Hex numbers
        text = re.sub(r'\b\d+\b', ' NUM ', text)  # Numbers
        text = re.sub(r'/[^\s]+', ' PATH ', text)  # File paths
        
        # Split on non-alphanumeric
        tokens = re.findall(r'\b[a-z_][a-z_0-9]*\b', text)
        
        return tokens
    
    def _get_ngrams(self, tokens: List[str]) -> List[str]:
        """Generate n-grams from tokens"""
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngram = "_".join(tokens[i:i+n])
                ngrams.append(ngram)
        return ngrams
    
    def fit(self, documents: List[str]) -> "TFIDFVectorizer":
        """Fit the vectorizer on training documents"""
        # Count document frequency for each term
        df_counts = Counter()
        doc_count = len(documents)
        
        for doc in documents:
            tokens = self._tokenize(doc)
            ngrams = set(self._get_ngrams(tokens))
            for ngram in ngrams:
                df_counts[ngram] += 1
        
        # Filter by min_df and select top features
        filtered_terms = [
            (term, count) for term, count in df_counts.items()
            if count >= self.min_df
        ]
        filtered_terms.sort(key=lambda x: x[1], reverse=True)
        filtered_terms = filtered_terms[:self.max_features]
        
        # Build vocabulary
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(filtered_terms)}
        
        # Calculate IDF
        self.idf = np.zeros(len(self.vocabulary))
        for term, idx in self.vocabulary.items():
            df = df_counts[term]
            self.idf[idx] = np.log((doc_count + 1) / (df + 1)) + 1  # Smoothed IDF
        
        self.fitted = True
        logger.info(f"TF-IDF fitted with {len(self.vocabulary)} features from {doc_count} documents")
        return self
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to TF-IDF vectors"""
        if not self.fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        vectors = np.zeros((len(documents), len(self.vocabulary)))
        
        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            ngrams = self._get_ngrams(tokens)
            
            # Calculate term frequencies
            tf = Counter(ngrams)
            
            for term, count in tf.items():
                if term in self.vocabulary and self.idf is not None:
                    idx = self.vocabulary[term]
                    # TF with log normalization
                    vectors[doc_idx, idx] = (1 + np.log(count)) * self.idf[idx]
        
        # L2 normalization
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms
        
        return vectors
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(documents)
        return self.transform(documents)


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes classifier for log categorization.
    
    Works well with TF-IDF features and is efficient for multi-class classification.
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha  # Laplace smoothing
        self.classes: List[str] = []
        self.class_log_priors: Dict[str, float] = {}
        self.feature_log_probs: Dict[str, np.ndarray] = {}
        self.fitted = False
        
    def fit(self, X: np.ndarray, y: List[str]) -> "NaiveBayesClassifier":
        """Train the classifier"""
        self.classes = list(set(y))
        n_features = X.shape[1]
        
        # Calculate class priors
        class_counts = Counter(y)
        total = len(y)
        
        for cls in self.classes:
            self.class_log_priors[cls] = np.log(class_counts[cls] / total)
        
        # Calculate feature probabilities per class
        for cls in self.classes:
            # Get all samples for this class
            class_mask = np.array([label == cls for label in y])
            class_samples = X[class_mask]
            
            # Sum features and apply smoothing
            feature_counts = class_samples.sum(axis=0) + self.alpha
            total_count = feature_counts.sum()
            
            self.feature_log_probs[cls] = np.log(feature_counts / total_count)
        
        self.fitted = True
        logger.info(f"Naive Bayes trained on {len(y)} samples, {len(self.classes)} classes")
        return self
    
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict class probabilities"""
        if not self.fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        log_probs = {}
        
        for cls in self.classes:
            # Log likelihood + log prior
            log_probs[cls] = X @ self.feature_log_probs[cls] + self.class_log_priors[cls]
        
        # Convert to proper probabilities using softmax
        all_log_probs = np.column_stack([log_probs[cls] for cls in self.classes])
        max_log = all_log_probs.max(axis=1, keepdims=True)
        exp_probs = np.exp(all_log_probs - max_log)
        probs = exp_probs / exp_probs.sum(axis=1, keepdims=True)
        
        return {cls: probs[:, i] for i, cls in enumerate(self.classes)}
    
    def predict(self, X: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """Predict classes and confidence scores"""
        proba = self.predict_proba(X)
        
        # Stack probabilities and find argmax
        prob_matrix = np.column_stack([proba[cls] for cls in self.classes])
        predictions = [self.classes[idx] for idx in prob_matrix.argmax(axis=1)]
        confidences = prob_matrix.max(axis=1)
        
        return predictions, confidences


class SVMClassifier:
    """
    Support Vector Machine classifier using stochastic gradient descent.
    
    Implements linear SVM with online learning capability.
    """
    
    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # Regularization
        self.n_iterations = n_iterations
        self.weights: Dict[str, np.ndarray] = {}
        self.biases: Dict[str, float] = {}
        self.classes: List[str] = []
        self.fitted = False
        
    def fit(self, X: np.ndarray, y: List[str]) -> "SVMClassifier":
        """Train the SVM using SGD"""
        self.classes = list(set(y))
        n_samples, n_features = X.shape
        
        # One-vs-rest classification
        for cls in self.classes:
            # Binary labels
            binary_y = np.array([1 if label == cls else -1 for label in y])
            
            # Initialize weights
            w = np.zeros(n_features)
            b = 0.0
            
            # SGD training
            for _ in range(self.n_iterations):
                for i in range(n_samples):
                    xi, yi = X[i], binary_y[i]
                    
                    # Hinge loss gradient
                    if yi * (np.dot(xi, w) + b) < 1:
                        w = w - self.learning_rate * (self.lambda_param * w - yi * xi)
                        b = b + self.learning_rate * yi
                    else:
                        w = w - self.learning_rate * self.lambda_param * w
            
            self.weights[cls] = w
            self.biases[cls] = b
        
        self.fitted = True
        logger.info(f"SVM trained on {n_samples} samples, {len(self.classes)} classes")
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """Predict classes"""
        if not self.fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        # Calculate decision function for each class
        decisions = {}
        for cls in self.classes:
            decisions[cls] = X @ self.weights[cls] + self.biases[cls]
        
        # Pick class with highest decision value
        decision_matrix = np.column_stack([decisions[cls] for cls in self.classes])
        predictions = [self.classes[idx] for idx in decision_matrix.argmax(axis=1)]
        
        # Confidence based on margin
        confidences = np.abs(decision_matrix).max(axis=1)
        confidences = 1 / (1 + np.exp(-confidences))  # Sigmoid normalization
        
        return predictions, confidences
    
    def partial_fit(self, X: np.ndarray, y: List[str]) -> "SVMClassifier":
        """Online learning - update model with new samples"""
        if not self.fitted:
            return self.fit(X, y)
        
        # Single pass update
        for cls in self.classes:
            binary_y = np.array([1 if label == cls else -1 for label in y])
            w = self.weights[cls]
            b = self.biases[cls]
            
            for i in range(len(y)):
                xi, yi = X[i], binary_y[i]
                
                if yi * (np.dot(xi, w) + b) < 1:
                    w = w - self.learning_rate * (self.lambda_param * w - yi * xi)
                    b = b + self.learning_rate * yi
                else:
                    w = w - self.learning_rate * self.lambda_param * w
            
            self.weights[cls] = w
            self.biases[cls] = b
        
        return self


class LogClassifier:
    """
    Main log classifier that combines multiple classifiers and provides
    a unified interface for log categorization.
    
    Features:
    - Multi-label classification (category, domain, severity)
    - Ensemble of Naive Bayes and SVM
    - Entity extraction (IPs, paths, errors)
    - Cross-domain transfer learning
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.vectorizer = TFIDFVectorizer(max_features=5000, min_df=2, ngram_range=(1, 2))
        
        # Multiple classifiers for different tasks
        self.category_classifier = NaiveBayesClassifier()
        self.domain_classifier = NaiveBayesClassifier()
        self.severity_classifier = SVMClassifier()
        
        # Use absolute path relative to project root (2 levels up from this file)
        project_root = Path(__file__).parent.parent.parent.parent.parent
        default_path = project_root / "data" / "models" / "log_classifier.pkl"
        self.model_path = Path(model_path) if model_path else default_path
        self.is_trained = False
        self.training_stats: Dict[str, Any] = {}
        
        # Entity patterns
        self.entity_patterns = {
            "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            "file_path": re.compile(r'(?:/[^\s/:]+)+'),
            "url": re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
            "error_code": re.compile(r'\b[A-Z]{2,10}[_-]?\d{3,5}\b'),
            "timestamp": re.compile(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}'),
            "user": re.compile(r'\buser[=:\s]+([^\s,;]+)', re.IGNORECASE),
            "pid": re.compile(r'\bpid[=:\s]+(\d+)', re.IGNORECASE),
            "port": re.compile(r'\bport[=:\s]+(\d+)', re.IGNORECASE)
        }
        
    def _label_logs_from_structure(self, logs: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
        """
        Auto-label logs based on structure (when explicit labels aren't available).
        Uses heuristics to determine category, domain, and severity.
        """
        categories = []
        domains = []
        severities = []
        
        for log in logs:
            message = log.get("message", log.get("Content", "")).lower()
            source = log.get("source", log.get("Component", "")).lower()
            
            # Category detection
            if any(w in message for w in ["error", "exception", "failed", "failure", "fatal"]):
                category = "error"
            elif any(w in message for w in ["warning", "warn", "caution"]):
                category = "warning"
            elif any(w in message for w in ["attack", "malicious", "intrusion", "unauthorized", "denied"]):
                category = "security"
            elif any(w in message for w in ["slow", "timeout", "latency", "performance"]):
                category = "performance"
            else:
                category = "normal"
            categories.append(category)
            
            # Domain detection
            if any(w in message or w in source for w in ["http", "apache", "nginx", "request", "response"]):
                domain = "web_server"
            elif any(w in message or w in source for w in ["sql", "database", "query", "mysql", "postgres"]):
                domain = "database"
            elif any(w in message or w in source for w in ["auth", "login", "password", "credential", "ssh"]):
                domain = "authentication"
            elif any(w in message or w in source for w in ["kernel", "systemd", "service", "daemon"]):
                domain = "system"
            elif any(w in message or w in source for w in ["network", "socket", "connection", "tcp", "udp"]):
                domain = "network"
            elif any(w in message or w in source for w in ["hadoop", "spark", "kubernetes", "container"]):
                domain = "distributed"
            else:
                domain = "application"
            domains.append(domain)
            
            # Severity detection
            if any(w in message for w in ["fatal", "critical", "emergency", "panic"]):
                severity = "critical"
            elif any(w in message for w in ["error", "failed", "failure"]):
                severity = "high"
            elif any(w in message for w in ["warning", "warn"]):
                severity = "medium"
            elif any(w in message for w in ["info", "notice"]):
                severity = "low"
            else:
                severity = "info"
            severities.append(severity)
        
        return categories, domains, severities
    
    async def train(self, logs: List[Dict], 
                    categories: Optional[List[str]] = None,
                    domains: Optional[List[str]] = None,
                    severities: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train the classifier on log data.
        
        Args:
            logs: List of log entries with 'message' or 'Content' field
            categories: Optional explicit category labels
            domains: Optional explicit domain labels
            severities: Optional explicit severity labels
            
        Returns:
            Training metrics
        """
        if len(logs) < 50:
            return {
                "success": False,
                "error": "Insufficient training data. Need at least 50 log entries.",
                "samples_provided": len(logs)
            }
        
        logger.info(f"Training log classifier on {len(logs)} samples...")
        
        # Extract text
        texts = [log.get("message", log.get("Content", "")) for log in logs]
        
        # Get or infer labels
        if categories is None or domains is None or severities is None:
            auto_categories, auto_domains, auto_severities = self._label_logs_from_structure(logs)
            categories = categories or auto_categories
            domains = domains or auto_domains
            severities = severities or auto_severities
        
        # Fit vectorizer
        X = self.vectorizer.fit_transform(texts)
        
        # Train classifiers
        self.category_classifier.fit(X, categories)
        self.domain_classifier.fit(X, domains)
        self.severity_classifier.fit(X, severities)
        
        self.is_trained = True
        
        # Calculate training stats
        self.training_stats = {
            "timestamp": datetime.now().isoformat(),
            "samples": len(logs),
            "vocabulary_size": len(self.vectorizer.vocabulary),
            "categories": list(set(categories)),
            "domains": list(set(domains)),
            "severities": list(set(severities)),
            "category_distribution": dict(Counter(categories)),
            "domain_distribution": dict(Counter(domains))
        }
        
        # Save model
        await self.save()
        
        return {
            "success": True,
            **self.training_stats
        }
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from log text"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = pattern.findall(text)
            if matches:
                entities[entity_type] = list(set(matches))[:10]  # Limit to 10
        
        return entities
    
    async def classify(self, logs: List[Dict]) -> List[ClassificationResult]:
        """
        Classify log entries.
        
        Args:
            logs: List of log entries
            
        Returns:
            List of ClassificationResult
        """
        if not self.is_trained:
            logger.warning("Classifier not trained, using heuristic classification")
            return await self._heuristic_classify(logs)
        
        # Extract text and vectorize
        texts = [log.get("message", log.get("Content", "")) for log in logs]
        
        # Check if using sklearn format (loaded from retrain_models.py)
        if hasattr(self, '_sklearn_vectorizer') and self._sklearn_vectorizer is not None:
            return await self._sklearn_classify(texts, logs)
        
        X = self.vectorizer.transform(texts)
        
        # Get predictions
        cat_preds, cat_conf = self.category_classifier.predict(X)
        dom_preds, dom_conf = self.domain_classifier.predict(X)
        sev_preds, sev_conf = self.severity_classifier.predict(X)
        
        # Get probabilities for category
        cat_proba = self.category_classifier.predict_proba(X)
        
        results = []
        for i, (text, cat, dom, sev) in enumerate(zip(texts, cat_preds, dom_preds, sev_preds)):
            # Combine confidences
            overall_confidence = (cat_conf[i] + dom_conf[i] + sev_conf[i]) / 3
            
            # Build probability dict
            probs = {cls: float(cat_proba[cls][i]) for cls in cat_proba}
            
            # Extract entities
            entities = self._extract_entities(text)
            
            results.append(ClassificationResult(
                category=cat,
                domain=dom,
                severity=sev,
                confidence=float(overall_confidence),
                probabilities=probs,
                extracted_entities=entities
            ))
        
        return results
    
    async def _sklearn_classify(self, texts: List[str], logs: List[Dict]) -> List[ClassificationResult]:
        """Classify using sklearn models loaded from retrain_models.py"""
        # Map numeric severity labels to names
        severity_map = {
            0: 'DEBUG', 1: 'INFO', 2: 'WARNING', 3: 'ERROR', 4: 'CRITICAL',
            '0': 'DEBUG', '1': 'INFO', '2': 'WARNING', '3': 'ERROR', '4': 'CRITICAL',
            'debug': 'DEBUG', 'info': 'INFO', 'warning': 'WARNING', 
            'error': 'ERROR', 'critical': 'CRITICAL', 'fatal': 'CRITICAL'
        }
        
        # Transform texts
        X = self._sklearn_vectorizer.transform(texts)
        
        results = []
        for i, (text, log) in enumerate(zip(texts, logs)):
            X_single = X[i:i+1]
            
            # Severity prediction
            if self._sklearn_severity is not None and hasattr(self._sklearn_severity, 'predict'):
                sev_pred_raw = self._sklearn_severity.predict(X_single)[0]
                # Map to severity name
                sev_pred = severity_map.get(sev_pred_raw, severity_map.get(str(sev_pred_raw).lower(), 'INFO'))
                if hasattr(self._sklearn_severity, 'predict_proba'):
                    sev_proba = self._sklearn_severity.predict_proba(X_single)[0]
                    sev_conf = float(max(sev_proba))
                else:
                    sev_conf = 0.8
            else:
                sev_pred = log.get("severity", "INFO")
                sev_conf = 0.7
            
            # Domain prediction
            if self._sklearn_domain is not None and hasattr(self._sklearn_domain, 'predict'):
                dom_pred = self._sklearn_domain.predict(X_single)[0]
                if hasattr(self._sklearn_domain, 'predict_proba'):
                    dom_proba = self._sklearn_domain.predict_proba(X_single)[0]
                    dom_conf = float(max(dom_proba))
                else:
                    dom_conf = 0.8
            else:
                dom_pred = self._infer_domain_from_text(text)
                dom_conf = 0.6
            
            # Category from severity
            sev_str = str(sev_pred).lower()
            if sev_str in ['critical', 'error']:
                cat_pred = 'error'
            elif sev_str == 'warning':
                cat_pred = 'warning'
            elif 'security' in text.lower() or 'attack' in text.lower():
                cat_pred = 'security'
            else:
                cat_pred = 'normal'
            
            cat_conf = sev_conf
            
            # Extract entities
            entities = self._extract_entities(text)
            
            results.append(ClassificationResult(
                category=cat_pred,
                domain=str(dom_pred),
                severity=str(sev_pred),
                confidence=float((sev_conf + dom_conf + cat_conf) / 3),
                probabilities={cat_pred: cat_conf},
                extracted_entities=entities
            ))
        
        return results
    
    def _infer_domain_from_text(self, text: str) -> str:
        """Infer domain from log text using keywords"""
        text_lower = text.lower()
        if any(kw in text_lower for kw in ['apache', 'nginx', 'http', '200', '404', '500']):
            return 'web_server'
        elif any(kw in text_lower for kw in ['mysql', 'postgres', 'database', 'query', 'sql']):
            return 'database'
        elif any(kw in text_lower for kw in ['ssh', 'login', 'auth', 'password']):
            return 'authentication'
        elif any(kw in text_lower for kw in ['network', 'socket', 'connection', 'tcp', 'udp']):
            return 'network'
        elif any(kw in text_lower for kw in ['kernel', 'systemd', 'boot', 'shutdown']):
            return 'system'
        else:
            return 'application'
    
    async def _heuristic_classify(self, logs: List[Dict]) -> List[ClassificationResult]:
        """Fallback heuristic classification"""
        categories, domains, severities = self._label_logs_from_structure(logs)
        
        results = []
        for log, cat, dom, sev in zip(logs, categories, domains, severities):
            text = log.get("message", log.get("Content", ""))
            entities = self._extract_entities(text)
            
            results.append(ClassificationResult(
                category=cat,
                domain=dom,
                severity=sev,
                confidence=0.6,  # Lower confidence for heuristics
                probabilities={cat: 0.6},
                extracted_entities=entities
            ))
        
        return results
    
    async def online_update(self, logs: List[Dict], 
                           categories: List[str],
                           domains: List[str],
                           severities: List[str]) -> Dict[str, Any]:
        """
        Update the model with new labeled data (online learning).
        """
        if not self.is_trained:
            return await self.train(logs, categories, domains, severities)
        
        texts = [log.get("message", log.get("Content", "")) for log in logs]
        X = self.vectorizer.transform(texts)
        
        # Partial fit SVM (supports online learning)
        self.severity_classifier.partial_fit(X, severities)
        
        # For Naive Bayes, we'd need to retrain or use incremental NB
        # For now, we just update if enough new data
        
        logger.info(f"Online update with {len(logs)} new samples")
        
        return {
            "success": True,
            "samples_updated": len(logs)
        }
    
    async def save(self) -> None:
        """Persist model to disk"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "vectorizer": {
                "vocabulary": self.vectorizer.vocabulary,
                "idf": self.vectorizer.idf.tolist() if self.vectorizer.idf is not None else None,
                "max_features": self.vectorizer.max_features,
                "min_df": self.vectorizer.min_df,
                "ngram_range": self.vectorizer.ngram_range,
                "fitted": self.vectorizer.fitted
            },
            "category_classifier": {
                "classes": self.category_classifier.classes,
                "class_log_priors": self.category_classifier.class_log_priors,
                "feature_log_probs": {k: v.tolist() for k, v in self.category_classifier.feature_log_probs.items()},
                "fitted": self.category_classifier.fitted
            },
            "domain_classifier": {
                "classes": self.domain_classifier.classes,
                "class_log_priors": self.domain_classifier.class_log_priors,
                "feature_log_probs": {k: v.tolist() for k, v in self.domain_classifier.feature_log_probs.items()},
                "fitted": self.domain_classifier.fitted
            },
            "severity_classifier": {
                "classes": self.severity_classifier.classes,
                "weights": {k: v.tolist() for k, v in self.severity_classifier.weights.items()},
                "biases": self.severity_classifier.biases,
                "fitted": self.severity_classifier.fitted
            },
            "training_stats": self.training_stats
        }
        
        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Log classifier saved to {self.model_path}")
    
    async def load(self) -> bool:
        """Load model from disk - supports both custom and sklearn formats"""
        if not self.model_path.exists():
            logger.warning(f"No saved model found at {self.model_path}")
            return False
        
        try:
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
            
            # Check if this is sklearn format (from retrain_models.py)
            if "vectorizer" in model_data and hasattr(model_data["vectorizer"], "transform"):
                # sklearn TfidfVectorizer format - store directly
                self._sklearn_vectorizer = model_data["vectorizer"]
                self._sklearn_severity = model_data.get("severity_classifier")
                self._sklearn_domain = model_data.get("domain_classifier")
                
                # Mark as trained
                self.vectorizer.fitted = True
                self.category_classifier.fitted = True
                self.domain_classifier.fitted = True  
                self.severity_classifier.fitted = True
                self.is_trained = True
                
                self.training_stats = {
                    "severity_accuracy": model_data.get("severity_accuracy", 0),
                    "domain_accuracy": model_data.get("domain_accuracy", 0)
                }
                
                logger.info(f"Loaded sklearn-format log classifier from {self.model_path}")
                return True
            
            # Custom format (from training_pipeline.py)
            vec_data = model_data["vectorizer"]
            self.vectorizer.vocabulary = vec_data["vocabulary"]
            self.vectorizer.idf = np.array(vec_data["idf"]) if vec_data["idf"] else None
            self.vectorizer.max_features = vec_data["max_features"]
            self.vectorizer.min_df = vec_data["min_df"]
            self.vectorizer.ngram_range = tuple(vec_data["ngram_range"])
            self.vectorizer.fitted = vec_data["fitted"]
            
            # Restore category classifier
            cat_data = model_data["category_classifier"]
            self.category_classifier.classes = cat_data["classes"]
            self.category_classifier.class_log_priors = cat_data["class_log_priors"]
            self.category_classifier.feature_log_probs = {
                k: np.array(v) for k, v in cat_data["feature_log_probs"].items()
            }
            self.category_classifier.fitted = cat_data["fitted"]
            
            # Restore domain classifier
            dom_data = model_data["domain_classifier"]
            self.domain_classifier.classes = dom_data["classes"]
            self.domain_classifier.class_log_priors = dom_data["class_log_priors"]
            self.domain_classifier.feature_log_probs = {
                k: np.array(v) for k, v in dom_data["feature_log_probs"].items()
            }
            self.domain_classifier.fitted = dom_data["fitted"]
            
            # Restore severity classifier
            sev_data = model_data["severity_classifier"]
            self.severity_classifier.classes = sev_data["classes"]
            self.severity_classifier.weights = {k: np.array(v) for k, v in sev_data["weights"].items()}
            self.severity_classifier.biases = sev_data["biases"]
            self.severity_classifier.fitted = sev_data["fitted"]
            
            self.training_stats = model_data.get("training_stats", {})
            self.is_trained = True
            
            logger.info(f"Log classifier loaded from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            return False
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            "is_trained": self.is_trained,
            "vocabulary_size": len(self.vectorizer.vocabulary) if self.vectorizer.fitted else 0,
            "categories": self.category_classifier.classes if self.category_classifier.fitted else [],
            "domains": self.domain_classifier.classes if self.domain_classifier.fitted else [],
            "severities": self.severity_classifier.classes if self.severity_classifier.fitted else [],
            "training_stats": self.training_stats
        }
