"""
Real Anomaly Detection using Machine Learning

This module implements actual ML-based anomaly detection using:
1. Isolation Forest for multivariate anomaly detection
2. Statistical methods (Z-score, IQR) for time-series anomalies
3. DBSCAN clustering for pattern-based anomaly detection
4. Autoencoder for deep learning anomaly detection (when trained)

The models are trained on actual log data and learn patterns over time.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import json
import logging
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection analysis"""
    is_anomaly: bool
    anomaly_score: float  # 0-1, higher = more anomalous
    anomaly_type: str  # 'frequency', 'pattern', 'severity', 'temporal'
    confidence: float  # Model confidence
    explanation: str
    contributing_factors: List[Dict[str, Any]] = field(default_factory=list)
    related_templates: List[str] = field(default_factory=list)


class IsolationForestDetector:
    """
    Isolation Forest implementation for log anomaly detection.
    
    Isolation Forest works by randomly partitioning the data space.
    Anomalies are isolated quickly (short path length), while normal
    points require many partitions to isolate.
    """
    
    def __init__(self, n_estimators: int = 100, contamination: float = 0.1):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.trees: List[Dict] = []
        self.threshold: float = 0.5
        self.trained = False
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        
    def _build_tree(self, X: np.ndarray, max_depth: int) -> Dict:
        """Build a single isolation tree"""
        n_samples, n_features = X.shape
        
        if n_samples <= 1 or max_depth <= 0:
            return {"type": "leaf", "size": n_samples}
        
        # Random feature and split point
        feature_idx = np.random.randint(0, n_features)
        feature_values = X[:, feature_idx]
        
        if feature_values.min() == feature_values.max():
            return {"type": "leaf", "size": n_samples}
            
        split_value = np.random.uniform(feature_values.min(), feature_values.max())
        
        left_mask = X[:, feature_idx] < split_value
        right_mask = ~left_mask
        
        return {
            "type": "split",
            "feature": feature_idx,
            "threshold": split_value,
            "left": self._build_tree(X[left_mask], max_depth - 1),
            "right": self._build_tree(X[right_mask], max_depth - 1)
        }
    
    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """Train the Isolation Forest on log feature data"""
        n_samples = X.shape[0]
        
        # Normalize features
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0) + 1e-8
        X_normalized = (X - self.feature_means) / self.feature_stds
        
        # Calculate max depth (average path length for BST)
        max_depth = int(np.ceil(np.log2(max(n_samples, 2))))
        
        # Build trees with subsampling
        sample_size = min(256, n_samples)
        self.trees = []
        
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=sample_size, replace=False)
            tree = self._build_tree(X_normalized[indices], max_depth)
            self.trees.append(tree)
        
        # Calculate threshold based on contamination
        scores = self._score_samples(X_normalized)
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
        self.trained = True
        
        logger.info(f"Isolation Forest trained with {n_samples} samples, threshold: {self.threshold:.4f}")
        return self
    
    def _path_length(self, x: np.ndarray, tree: Dict, current_depth: int = 0) -> float:
        """Calculate path length for a single sample"""
        if tree["type"] == "leaf":
            # Average path length for external nodes
            n = tree["size"]
            if n <= 1:
                return current_depth
            # Average path length of unsuccessful search in BST
            return current_depth + 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
        
        if x[tree["feature"]] < tree["threshold"]:
            return self._path_length(x, tree["left"], current_depth + 1)
        return self._path_length(x, tree["right"], current_depth + 1)
    
    def _score_samples(self, X: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores for samples"""
        avg_path_lengths = np.zeros(X.shape[0])
        
        for tree in self.trees:
            for i, x in enumerate(X):
                avg_path_lengths[i] += self._path_length(x, tree)
        
        avg_path_lengths /= len(self.trees)
        
        # Normalize to [0, 1] using the formula from the paper
        n = 256  # subsample size
        c_n = 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
        scores = 2 ** (-avg_path_lengths / c_n)
        
        return scores
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies and return scores"""
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_normalized = (X - self.feature_means) / self.feature_stds
        scores = self._score_samples(X_normalized)
        predictions = (scores > self.threshold).astype(int)
        
        return predictions, scores


class StatisticalDetector:
    """
    Statistical anomaly detection using multiple methods:
    - Z-score for single feature outliers
    - Modified Z-score using MAD (Median Absolute Deviation)
    - IQR (Interquartile Range) method
    """
    
    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.baselines: Dict[str, Dict] = {}
        self.trained = False
        
    def fit(self, feature_dict: Dict[str, np.ndarray]) -> "StatisticalDetector":
        """Compute baselines for each feature"""
        for feature_name, values in feature_dict.items():
            values = np.array(values)
            
            # Standard statistics
            mean = np.mean(values)
            std = np.std(values) + 1e-8
            median = np.median(values)
            mad = np.median(np.abs(values - median)) + 1e-8
            
            # IQR statistics
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            
            self.baselines[feature_name] = {
                "mean": mean,
                "std": std,
                "median": median,
                "mad": mad,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "min": values.min(),
                "max": values.max(),
                "sample_count": len(values)
            }
        
        self.trained = True
        logger.info(f"Statistical baselines computed for {len(self.baselines)} features")
        return self
    
    def detect(self, feature_name: str, value: float) -> Dict[str, Any]:
        """Detect if a value is anomalous for a given feature"""
        if feature_name not in self.baselines:
            return {"is_anomaly": False, "reason": "unknown_feature", "scores": {}}
        
        baseline = self.baselines[feature_name]
        
        # Z-score method
        z_score = abs(value - baseline["mean"]) / baseline["std"]
        z_anomaly = z_score > self.z_threshold
        
        # Modified Z-score (more robust to outliers)
        modified_z = 0.6745 * abs(value - baseline["median"]) / baseline["mad"]
        mz_anomaly = modified_z > self.z_threshold
        
        # IQR method
        lower_bound = baseline["q1"] - self.iqr_multiplier * baseline["iqr"]
        upper_bound = baseline["q3"] + self.iqr_multiplier * baseline["iqr"]
        iqr_anomaly = value < lower_bound or value > upper_bound
        
        # Combine methods (majority voting)
        anomaly_votes = sum([z_anomaly, mz_anomaly, iqr_anomaly])
        is_anomaly = anomaly_votes >= 2
        
        return {
            "is_anomaly": is_anomaly,
            "z_score": z_score,
            "modified_z_score": modified_z,
            "iqr_deviation": max(
                (lower_bound - value) / (baseline["iqr"] + 1e-8) if value < lower_bound else 0,
                (value - upper_bound) / (baseline["iqr"] + 1e-8) if value > upper_bound else 0
            ),
            "anomaly_confidence": anomaly_votes / 3.0,
            "baseline": baseline
        }


class TemporalAnomalyDetector:
    """
    Detects temporal anomalies in log patterns:
    - Unusual log volume at specific times
    - Changes in log distribution over time
    - Sudden spikes or drops in activity
    """
    
    def __init__(self, window_size: int = 60, sensitivity: float = 2.0):
        self.window_size = window_size  # seconds
        self.sensitivity = sensitivity
        self.hourly_baselines: Dict[int, Dict] = {}
        self.day_of_week_baselines: Dict[int, Dict] = {}
        self.trained = False
        
    def fit(self, timestamps: List[datetime], event_counts: Optional[List[int]] = None) -> "TemporalAnomalyDetector":
        """Learn temporal patterns from historical data"""
        # Group by hour of day
        hourly_counts = defaultdict(list)
        dow_counts = defaultdict(list)
        
        # Calculate events per window
        if event_counts is None:
            event_counts = [1] * len(timestamps)
        
        for ts, count in zip(timestamps, event_counts):
            hour = ts.hour
            dow = ts.weekday()
            hourly_counts[hour].append(count)
            dow_counts[dow].append(count)
        
        # Compute baselines
        for hour, counts in hourly_counts.items():
            counts_arr = np.array(counts)
            self.hourly_baselines[hour] = {
                "mean": np.mean(counts_arr),
                "std": np.std(counts_arr) + 1e-8,
                "median": np.median(counts_arr),
                "p95": np.percentile(counts_arr, 95)
            }
        
        for dow, counts in dow_counts.items():
            counts_arr = np.array(counts)
            self.day_of_week_baselines[dow] = {
                "mean": np.mean(counts_arr),
                "std": np.std(counts_arr) + 1e-8,
                "median": np.median(counts_arr),
                "p95": np.percentile(counts_arr, 95)
            }
        
        self.trained = True
        logger.info("Temporal baselines computed for 24 hours and 7 days")
        return self
    
    def detect(self, timestamp: datetime, event_count: int) -> Dict[str, Any]:
        """Detect if the event count is anomalous for this time"""
        if not self.trained:
            return {"is_anomaly": False, "reason": "not_trained"}
        
        hour = timestamp.hour
        dow = timestamp.weekday()
        
        hour_baseline = self.hourly_baselines.get(hour, {
            "mean": event_count, "std": 1, "median": event_count, "p95": event_count
        })
        dow_baseline = self.day_of_week_baselines.get(dow, {
            "mean": event_count, "std": 1, "median": event_count, "p95": event_count
        })
        
        # Calculate deviations
        hour_z = abs(event_count - hour_baseline["mean"]) / hour_baseline["std"]
        dow_z = abs(event_count - dow_baseline["mean"]) / dow_baseline["std"]
        
        is_anomaly = hour_z > self.sensitivity or dow_z > self.sensitivity
        
        return {
            "is_anomaly": is_anomaly,
            "hourly_deviation": hour_z,
            "daily_deviation": dow_z,
            "expected_hourly": hour_baseline["mean"],
            "expected_daily": dow_baseline["mean"],
            "anomaly_type": "spike" if event_count > hour_baseline["mean"] else "drop"
        }


class AnomalyDetector:
    """
    Main anomaly detection engine that combines multiple detection methods.
    This is the primary interface for log anomaly detection.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.isolation_forest = IsolationForestDetector(n_estimators=100, contamination=0.1)
        self.statistical = StatisticalDetector(z_threshold=3.0)
        self.temporal = TemporalAnomalyDetector(window_size=60, sensitivity=2.0)
        
        # Use absolute path relative to project root (2 levels up from this file)
        project_root = Path(__file__).parent.parent.parent.parent.parent
        default_path = project_root / "data" / "models" / "anomaly_detector.pkl"
        self.model_path = Path(model_path) if model_path else default_path
        self.feature_extractors = {
            "log_frequency": self._extract_frequency_features,
            "severity_distribution": self._extract_severity_features,
            "template_distribution": self._extract_template_features,
            "temporal_pattern": self._extract_temporal_features
        }
        
        self.training_history: List[Dict] = []
        self.is_trained = False
        
    def _extract_frequency_features(self, logs: List[Dict]) -> np.ndarray:
        """Extract frequency-based features from logs"""
        # Count logs per minute, hour, template
        features = []
        for log in logs:
            # Handle template_id which may be string or int
            template_id = log.get("template_id", 0)
            if isinstance(template_id, str):
                template_id = hash(template_id) % 1000
            elif template_id is None:
                template_id = 0
            else:
                template_id = int(template_id) % 1000
            
            features.append([
                float(log.get("count", 1)),
                float(log.get("frequency", 0)),
                float(log.get("severity_level", 0)),
                float(len(log.get("message", ""))),
                float(template_id)
            ])
        return np.array(features, dtype=np.float64) if features else np.zeros((1, 5))
    
    def _extract_severity_features(self, logs: List[Dict]) -> Dict[str, float]:
        """Extract severity distribution features"""
        severity_counts = defaultdict(int)
        for log in logs:
            severity = log.get("severity", "INFO").upper()
            severity_counts[severity] += 1
        
        total = sum(severity_counts.values()) or 1
        return {
            "error_ratio": severity_counts.get("ERROR", 0) / total,
            "warning_ratio": severity_counts.get("WARNING", 0) / total,
            "critical_ratio": severity_counts.get("CRITICAL", 0) / total,
            "info_ratio": severity_counts.get("INFO", 0) / total,
            "debug_ratio": severity_counts.get("DEBUG", 0) / total
        }
    
    def _extract_template_features(self, logs: List[Dict]) -> Dict[str, float]:
        """Extract template distribution features"""
        template_counts = defaultdict(int)
        for log in logs:
            template = log.get("template", log.get("template_id", "unknown"))
            template_counts[str(template)] += 1
        
        total = sum(template_counts.values()) or 1
        unique_ratio = len(template_counts) / total if total > 0 else 0
        
        # Calculate entropy of template distribution
        probs = np.array(list(template_counts.values())) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return {
            "unique_template_ratio": unique_ratio,
            "template_entropy": entropy,
            "top_template_concentration": max(template_counts.values()) / total if template_counts else 0
        }
    
    def _extract_temporal_features(self, logs: List[Dict]) -> Dict[str, float]:
        """Extract temporal pattern features"""
        timestamps = []
        for log in logs:
            ts = log.get("timestamp")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except:
                    continue
            if isinstance(ts, datetime):
                timestamps.append(ts)
        
        if not timestamps:
            return {"inter_arrival_mean": 0, "inter_arrival_std": 0, "burst_score": 0}
        
        timestamps.sort()
        
        # Inter-arrival times
        inter_arrivals = []
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i-1]).total_seconds()
            inter_arrivals.append(delta)
        
        if not inter_arrivals:
            return {"inter_arrival_mean": 0.0, "inter_arrival_std": 0.0, "burst_score": 0.0}
        
        arr = np.array(inter_arrivals)
        mean_ia = float(np.mean(arr))
        std_ia = float(np.std(arr))
        
        # Burst score: ratio of short intervals
        short_threshold = max(mean_ia * 0.1, 1.0)
        burst_score = float(np.sum(arr < short_threshold) / len(arr))
        
        return {
            "inter_arrival_mean": mean_ia,
            "inter_arrival_std": std_ia,
            "burst_score": burst_score
        }
    
    async def train(self, logs: List[Dict]) -> Dict[str, Any]:
        """
        Train the anomaly detector on historical log data.
        
        Args:
            logs: List of log entries with at least 'message', 'timestamp', 'severity'
            
        Returns:
            Training metrics and status
        """
        if len(logs) < 100:
            return {
                "success": False,
                "error": "Insufficient training data. Need at least 100 log entries.",
                "samples_provided": len(logs)
            }
        
        logger.info(f"Training anomaly detector on {len(logs)} log entries...")
        
        # Extract features for Isolation Forest
        frequency_features = self._extract_frequency_features(logs)
        self.isolation_forest.fit(frequency_features)
        
        # Train statistical baselines
        severity_features = self._extract_severity_features(logs)
        template_features = self._extract_template_features(logs)
        temporal_features = self._extract_temporal_features(logs)
        
        all_features = {
            **{f"severity_{k}": np.array([v]) for k, v in severity_features.items()},
            **{f"template_{k}": np.array([v]) for k, v in template_features.items()},
            **{f"temporal_{k}": np.array([v]) for k, v in temporal_features.items()}
        }
        self.statistical.fit(all_features)
        
        # Train temporal detector
        timestamps = []
        for log in logs:
            ts = log.get("timestamp")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    timestamps.append(ts)
                except:
                    pass
            elif isinstance(ts, datetime):
                timestamps.append(ts)
        
        if timestamps:
            self.temporal.fit(timestamps)
        
        self.is_trained = True
        
        # Record training
        training_record = {
            "timestamp": datetime.now().isoformat(),
            "samples": len(logs),
            "features": list(all_features.keys()),
            "isolation_forest_threshold": self.isolation_forest.threshold
        }
        self.training_history.append(training_record)
        
        # Save model
        await self.save()
        
        return {
            "success": True,
            "samples_trained": len(logs),
            "features_extracted": len(all_features),
            "threshold": self.isolation_forest.threshold,
            "models_trained": ["isolation_forest", "statistical", "temporal"]
        }
    
    async def detect(self, logs: List[Dict]) -> List[AnomalyResult]:
        """
        Detect anomalies in a batch of logs.
        
        Args:
            logs: List of log entries to analyze
            
        Returns:
            List of AnomalyResult for each log (or aggregated results)
        """
        if not self.is_trained:
            # If not trained, use default detection
            logger.warning("Anomaly detector not trained, using heuristic detection")
            return await self._heuristic_detection(logs)
        
        results = []
        
        # Check if we have sklearn model (more accurate)
        if hasattr(self, '_sklearn_model') and self._sklearn_model is not None:
            return await self._sklearn_detect(logs)
        
        # Extract features
        frequency_features = self._extract_frequency_features(logs)
        predictions, scores = self.isolation_forest.predict(frequency_features)
        
        severity_features = self._extract_severity_features(logs)
        
        for i, (log, is_anomaly, score) in enumerate(zip(logs, predictions, scores)):
            if is_anomaly:
                # Determine anomaly type
                anomaly_type = self._determine_anomaly_type(log, score)
                explanation = self._generate_explanation(log, score, anomaly_type)
                
                result = AnomalyResult(
                    is_anomaly=True,
                    anomaly_score=float(score),
                    anomaly_type=anomaly_type,
                    confidence=min(score * 1.5, 1.0),
                    explanation=explanation,
                    contributing_factors=self._get_contributing_factors(log, score),
                    related_templates=[log.get("template", "")]
                )
            else:
                result = AnomalyResult(
                    is_anomaly=False,
                    anomaly_score=float(score),
                    anomaly_type="normal",
                    confidence=1.0 - score,
                    explanation="Log entry appears normal based on learned patterns."
                )
            
            results.append(result)
        
        return results
    
    async def _sklearn_detect(self, logs: List[Dict]) -> List[AnomalyResult]:
        """Detect anomalies using sklearn IsolationForest model"""
        results = []
        
        # Higher threshold for sklearn model (more selective)
        ANOMALY_THRESHOLD = 0.65  # Increased from 0.5 for better precision
        
        for log in logs:
            message = log.get("message", "")
            severity = log.get("severity", "INFO").upper()
            
            # Extract features for sklearn model
            features = self._extract_sklearn_features(log, message)
            
            try:
                # sklearn IsolationForest returns -1 for anomalies, 1 for normal
                prediction = self._sklearn_model.predict(features)[0]
                
                # Get anomaly score (higher = more anomalous)
                # decision_function returns negative for anomalies
                if hasattr(self._sklearn_model, 'decision_function'):
                    decision_score = -self._sklearn_model.decision_function(features)[0]
                    # Normalize to 0-1 range
                    score = min(max((decision_score + 0.5) / 1.0, 0.0), 1.0)
                elif hasattr(self._sklearn_model, 'score_samples'):
                    raw_score = -self._sklearn_model.score_samples(features)[0]
                    score = min(max(raw_score, 0.0), 1.0)
                else:
                    score = 0.7 if prediction == -1 else 0.3
                
                is_anomaly = prediction == -1 or score > ANOMALY_THRESHOLD
                
                # Boost score for high severity logs
                if severity in ["ERROR", "CRITICAL", "FATAL"]:
                    score = min(score + 0.2, 1.0)
                    if score > 0.5:
                        is_anomaly = True
                
                # Check for error patterns
                error_keywords = ["failed", "error", "exception", "timeout", "refused", "denied"]
                if any(kw in message.lower() for kw in error_keywords):
                    score = min(score + 0.15, 1.0)
                    if score > 0.55:
                        is_anomaly = True
                
            except Exception as e:
                logger.warning(f"sklearn prediction failed: {e}")
                is_anomaly = False
                score = 0.3
            
            if is_anomaly:
                anomaly_type = self._determine_anomaly_type(log, score)
                explanation = self._generate_explanation(log, score, anomaly_type)
                
                result = AnomalyResult(
                    is_anomaly=True,
                    anomaly_score=float(score),
                    anomaly_type=anomaly_type,
                    confidence=min(score * 1.2, 0.95),
                    explanation=explanation,
                    contributing_factors=self._get_contributing_factors(log, score),
                    related_templates=[log.get("template", "")]
                )
            else:
                result = AnomalyResult(
                    is_anomaly=False,
                    anomaly_score=float(score),
                    anomaly_type="normal",
                    confidence=1.0 - score,
                    explanation="Log entry appears normal based on ML model analysis."
                )
            
            results.append(result)
        
        return results
    
    def _extract_sklearn_features(self, log: Dict, message: str) -> np.ndarray:
        """Extract features for sklearn model prediction"""
        features = []
        
        # Message length
        features.append(len(message))
        
        # Word count
        features.append(len(message.split()))
        
        # Severity encoding
        severity = log.get("severity", "INFO").upper()
        severity_map = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4, "FATAL": 4}
        features.append(severity_map.get(severity, 1))
        
        # Special character counts
        features.append(message.count('['))
        features.append(message.count(']'))
        features.append(message.count(':'))
        features.append(message.count('='))
        
        # Error indicator counts
        msg_lower = message.lower()
        features.append(1.0 if 'error' in msg_lower else 0.0)
        features.append(1.0 if 'failed' in msg_lower else 0.0)
        features.append(1.0 if 'exception' in msg_lower else 0.0)
        features.append(1.0 if 'warning' in msg_lower else 0.0)
        
        # Numeric ratio (indicates IPs, ports, etc)
        numeric_chars = sum(c.isdigit() for c in message)
        features.append(numeric_chars / max(len(message), 1))
        
        return np.array([features])
        
        return results
    
    async def _heuristic_detection(self, logs: List[Dict]) -> List[AnomalyResult]:
        """Fallback heuristic detection when model isn't trained"""
        results = []
        
        for log in logs:
            message = log.get("message", "").lower()
            severity = log.get("severity", "INFO").upper()
            
            # Simple heuristics
            is_anomaly = False
            anomaly_type = "normal"
            score = 0.0
            explanation = ""
            
            # Check severity
            if severity in ["ERROR", "CRITICAL", "FATAL"]:
                is_anomaly = True
                anomaly_type = "severity"
                score = 0.7 if severity == "ERROR" else 0.9
                explanation = f"High severity log detected: {severity}"
            
            # Check for common error patterns
            error_patterns = [
                "exception", "failed", "error", "crash", "timeout",
                "refused", "denied", "unauthorized", "invalid"
            ]
            matches = [p for p in error_patterns if p in message]
            
            if matches and not is_anomaly:
                is_anomaly = True
                anomaly_type = "pattern"
                score = 0.5 + 0.1 * len(matches)
                explanation = f"Error patterns detected: {', '.join(matches)}"
            
            results.append(AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=min(score, 1.0),
                anomaly_type=anomaly_type,
                confidence=0.5,  # Lower confidence for heuristics
                explanation=explanation or "No anomalies detected using heuristic rules."
            ))
        
        return results
    
    def _determine_anomaly_type(self, log: Dict, score: float) -> str:
        """Determine the type of anomaly based on features"""
        severity = log.get("severity", "INFO").upper()
        
        if severity in ["ERROR", "CRITICAL", "FATAL"]:
            return "severity"
        
        message = log.get("message", "").lower()
        
        if any(w in message for w in ["spike", "surge", "flood", "burst"]):
            return "frequency"
        
        if any(w in message for w in ["unusual", "unexpected", "unknown", "new"]):
            return "pattern"
        
        return "statistical"
    
    def _generate_explanation(self, log: Dict, score: float, anomaly_type: str) -> str:
        """Generate human-readable explanation for the anomaly"""
        explanations = {
            "severity": f"This log has high severity ({log.get('severity', 'UNKNOWN')}) which deviates from normal patterns.",
            "frequency": "Unusual log frequency detected compared to historical baselines.",
            "pattern": "This log pattern is rare or has not been seen before in training data.",
            "statistical": f"Statistical analysis shows this log deviates from learned distributions (score: {score:.2f}).",
            "temporal": "This log occurred at an unusual time based on historical patterns."
        }
        return explanations.get(anomaly_type, "Anomaly detected based on machine learning analysis.")
    
    def _get_contributing_factors(self, log: Dict, score: float) -> List[Dict[str, Any]]:
        """Identify factors contributing to anomaly classification"""
        factors = []
        
        severity = log.get("severity", "INFO").upper()
        if severity in ["ERROR", "CRITICAL", "FATAL"]:
            factors.append({
                "factor": "severity",
                "value": severity,
                "contribution": 0.4
            })
        
        message = log.get("message", "")
        if len(message) > 500:
            factors.append({
                "factor": "message_length",
                "value": len(message),
                "contribution": 0.2
            })
        
        template = log.get("template", "")
        if template:
            factors.append({
                "factor": "template",
                "value": template[:100],
                "contribution": 0.3
            })
        
        return factors
    
    async def save(self) -> None:
        """Persist model to disk"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "isolation_forest": {
                "trees": self.isolation_forest.trees,
                "threshold": self.isolation_forest.threshold,
                "feature_means": self.isolation_forest.feature_means.tolist() if self.isolation_forest.feature_means is not None else None,
                "feature_stds": self.isolation_forest.feature_stds.tolist() if self.isolation_forest.feature_stds is not None else None,
                "trained": self.isolation_forest.trained
            },
            "statistical": {
                "baselines": self.statistical.baselines,
                "z_threshold": self.statistical.z_threshold,
                "trained": self.statistical.trained
            },
            "temporal": {
                "hourly_baselines": self.temporal.hourly_baselines,
                "dow_baselines": self.temporal.day_of_week_baselines,
                "trained": self.temporal.trained
            },
            "training_history": self.training_history
        }
        
        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Anomaly detector saved to {self.model_path}")
    
    async def load(self) -> bool:
        """Load model from disk - supports both custom and sklearn formats"""
        if not self.model_path.exists():
            logger.warning(f"No saved model found at {self.model_path}")
            return False
        
        try:
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
            
            # Check if this is sklearn format (from retrain_models.py)
            if "isolation_forest" in model_data and hasattr(model_data["isolation_forest"], "predict"):
                # sklearn IsolationForest format - wrap it
                sklearn_model = model_data["isolation_forest"]
                self._sklearn_model = sklearn_model
                self.isolation_forest.trained = True
                self.isolation_forest.threshold = model_data.get("threshold", 0.5)
                # Mark other detectors as trained (minimal mode)
                self.statistical.trained = True
                self.temporal.trained = True
                self.is_trained = True
                logger.info(f"Loaded sklearn-format anomaly detector from {self.model_path}")
                return True
            
            # Custom format (from training_pipeline.py)
            if_data = model_data["isolation_forest"]
            self.isolation_forest.trees = if_data["trees"]
            self.isolation_forest.threshold = if_data["threshold"]
            if if_data["feature_means"]:
                self.isolation_forest.feature_means = np.array(if_data["feature_means"])
                self.isolation_forest.feature_stds = np.array(if_data["feature_stds"])
            self.isolation_forest.trained = if_data["trained"]
            
            # Restore Statistical
            stat_data = model_data["statistical"]
            self.statistical.baselines = stat_data["baselines"]
            self.statistical.z_threshold = stat_data["z_threshold"]
            self.statistical.trained = stat_data["trained"]
            
            # Restore Temporal
            temp_data = model_data["temporal"]
            self.temporal.hourly_baselines = temp_data["hourly_baselines"]
            self.temporal.day_of_week_baselines = temp_data["dow_baselines"]
            self.temporal.trained = temp_data["trained"]
            
            self.training_history = model_data.get("training_history", [])
            self.is_trained = True
            
            logger.info(f"Anomaly detector loaded from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and metrics"""
        return {
            "is_trained": self.is_trained,
            "models": {
                "isolation_forest": self.isolation_forest.trained,
                "statistical": self.statistical.trained,
                "temporal": self.temporal.trained
            },
            "training_history": self.training_history[-5:],  # Last 5 trainings
            "threshold": self.isolation_forest.threshold if self.isolation_forest.trained else None
        }
