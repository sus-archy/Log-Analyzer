"""
Training Pipeline for LogMind AI

This module provides a complete training pipeline that:
1. Loads log data from the Logs/loghub directory
2. Preprocesses and labels the data
3. Trains all ML models (anomaly, classifier, security)
4. Evaluates model performance
5. Saves trained models for production use

The pipeline can be run automatically on startup or triggered via API.
"""

import asyncio
import logging
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import random

from .anomaly_detector import AnomalyDetector
from .log_classifier import LogClassifier
from .security_threat_detector import SecurityThreatDetector
from ..core.config import settings

logger = logging.getLogger(__name__)


class LogDataLoader:
    """
    Loads and preprocesses log data from various sources.
    
    Supports:
    - Database (primary - fastest)
    - CSV files from LogHub dataset
    - Plain text log files
    """
    
    def __init__(self, logs_directory: str = "Logs/loghub"):
        self.logs_dir = Path(logs_directory)
        
        # Mapping of subdirectories to domains
        self.domain_mapping = {
            "Android": "mobile",
            "Apache": "web_server",
            "BGL": "system",
            "Hadoop": "distributed",
            "HDFS": "distributed",
            "HealthApp": "application",
            "HPC": "system",
            "Linux": "system",
            "Mac": "system",
            "OpenSSH": "authentication",
            "OpenStack": "distributed",
            "Proxifier": "network",
            "Spark": "distributed",
            "Thunderbird": "system",
            "Windows": "system",
            "Zookeeper": "distributed"
        }
    
    async def load_from_database(self, max_logs: int = 100000) -> List[Dict[str, Any]]:
        """
        Load logs directly from the SQLite database - FASTEST method.
        
        This uses the already-ingested data which includes all the large files.
        """
        from ..storage.db import get_db
        
        logger.info(f"Loading up to {max_logs:,} logs from database...")
        
        db = await get_db()
        
        # Get services with most logs (not just first 50)
        cursor = await db.execute(
            """
            SELECT service_name, COUNT(*) as cnt 
            FROM logs_stream 
            GROUP BY service_name 
            ORDER BY cnt DESC 
            LIMIT 30
            """
        )
        services = [(row[0], row[1]) for row in await cursor.fetchall()]
        logger.info(f"Found {len(services)} services with data")
        
        all_logs = []
        
        # Calculate per-service limit based on total
        total_available = sum(cnt for _, cnt in services)
        
        for service, count in services:
            # Proportional sampling - more logs from larger sources
            per_service = min(count, max(1000, max_logs * count // total_available))
            
            cursor = await db.execute(
                """
                SELECT 
                    id, timestamp_utc, service_name, severity, body_raw,
                    template_hash, parameters_json, tenant_id
                FROM logs_stream 
                WHERE service_name = ?
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (service, per_service)
            )
            rows = await cursor.fetchall()
            
            for row in rows:
                # Map severity string to level - handle int or string
                severity = row[3]
                if severity is None:
                    severity = "INFO"
                    severity_level = 1
                elif isinstance(severity, int):
                    severity_level = severity
                    severity_names = {0: "DEBUG", 1: "INFO", 2: "WARNING", 3: "ERROR", 4: "CRITICAL"}
                    severity = severity_names.get(severity, "INFO")
                else:
                    severity = str(severity)
                    severity_map = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "WARN": 2, "ERROR": 3, "CRITICAL": 4, "FATAL": 4}
                    severity_level = severity_map.get(severity.upper(), 1)
                
                domain = self._infer_domain(service)
                
                all_logs.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "source": row[2],
                    "service_name": row[2],
                    "severity": severity,
                    "severity_level": severity_level,
                    "message": row[4] or "",
                    "template_id": row[5],
                    "template_hash": row[5],
                    "domain": domain,
                })
            
            logger.info(f"Loaded {len(rows):,} logs from {service} (of {count:,})")
        
        logger.info(f"Total logs loaded from database: {len(all_logs):,}")
        return all_logs
    
    def _infer_domain(self, service_name: str) -> str:
        """Infer domain from service name"""
        service_lower = service_name.lower()
        for key, domain in self.domain_mapping.items():
            if key.lower() in service_lower:
                return domain
        return "application"
        
    async def load_all_logs(self, max_per_source: int = 2000) -> List[Dict[str, Any]]:
        """Load logs from all available sources (prefers database)"""
        # Try database first - it's much faster and has all data
        try:
            db_logs = await self.load_from_database(max_logs=max_per_source * 20)
            if len(db_logs) >= 1000:
                return db_logs
        except Exception as e:
            logger.warning(f"Could not load from database: {e}, falling back to files")
        
        # Fallback to files
        all_logs = []
        
        if not self.logs_dir.exists():
            logger.warning(f"Logs directory not found: {self.logs_dir}")
            return []
        
        # Load from subdirectories
        for subdir in self.logs_dir.iterdir():
            if subdir.is_dir():
                domain = self.domain_mapping.get(subdir.name, "application")
                logs = await self._load_from_directory(subdir, domain, max_per_source)
                all_logs.extend(logs)
                logger.info(f"Loaded {len(logs)} logs from {subdir.name}")
        
        # Load standalone files
        for file in self.logs_dir.glob("*.txt"):
            logs = await self._load_text_file(file, "application")
            all_logs.extend(logs[:max_per_source])
        
        logger.info(f"Total logs loaded: {len(all_logs)}")
        return all_logs
    
    async def _load_from_directory(self, directory: Path, domain: str, max_logs: int) -> List[Dict]:
        """Load logs from a subdirectory"""
        logs = []
        
        # Look for structured CSV files
        for csv_file in directory.glob("*_structured.csv"):
            csv_logs = await self._load_csv_file(csv_file, domain)
            logs.extend(csv_logs)
        
        # Look for text files
        for txt_file in directory.glob("*.log"):
            txt_logs = await self._load_text_file(txt_file, domain)
            logs.extend(txt_logs)
        
        # Limit and shuffle
        if len(logs) > max_logs:
            random.shuffle(logs)
            logs = logs[:max_logs]
        
        return logs
    
    async def _load_csv_file(self, file_path: Path, domain: str) -> List[Dict]:
        """Load logs from a structured CSV file"""
        logs = []
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    log_entry = {
                        "message": row.get("Content", row.get("Message", "")),
                        "timestamp": row.get("Time", row.get("Timestamp", datetime.now().isoformat())),
                        "source": row.get("Component", row.get("Source", file_path.parent.name)),
                        "domain": domain,
                        "template": row.get("EventTemplate", ""),
                        "template_id": row.get("EventId", ""),
                        "severity": self._infer_severity(row.get("Content", "")),
                        "original_file": str(file_path),
                        "line_id": row.get("LineId", len(logs))
                    }
                    logs.append(log_entry)
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
        
        return logs
    
    async def _load_text_file(self, file_path: Path, domain: str) -> List[Dict]:
        """Load logs from a plain text file"""
        logs = []
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    log_entry = {
                        "message": line,
                        "timestamp": datetime.now().isoformat(),
                        "source": file_path.stem,
                        "domain": domain,
                        "severity": self._infer_severity(line),
                        "original_file": str(file_path),
                        "line_id": i
                    }
                    logs.append(log_entry)
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
        
        return logs
    
    def _infer_severity(self, message: str) -> str:
        """Infer severity level from log message"""
        message_lower = message.lower()
        
        if any(w in message_lower for w in ["fatal", "critical", "emergency", "panic"]):
            return "CRITICAL"
        elif any(w in message_lower for w in ["error", "failed", "failure", "exception"]):
            return "ERROR"
        elif any(w in message_lower for w in ["warning", "warn"]):
            return "WARNING"
        elif any(w in message_lower for w in ["debug"]):
            return "DEBUG"
        else:
            return "INFO"


class DataPreprocessor:
    """
    Preprocesses log data for ML training.
    
    Tasks:
    - Cleaning and normalization
    - Label inference
    - Train/test splitting
    - Feature engineering
    """
    
    def __init__(self):
        self.severity_levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
        
    def preprocess(self, logs: List[Dict]) -> List[Dict]:
        """Clean and normalize log data"""
        processed = []
        
        for log in logs:
            processed_log = log.copy()
            
            # Clean message
            message = processed_log.get("message", "")
            message = self._clean_message(message)
            processed_log["message"] = message
            
            # Normalize severity
            severity = processed_log.get("severity", "INFO").upper()
            if severity not in self.severity_levels:
                severity = "INFO"
            processed_log["severity"] = severity
            
            # Add derived features
            processed_log["message_length"] = len(message)
            processed_log["word_count"] = len(message.split())
            processed_log["has_error_keywords"] = self._has_error_keywords(message)
            processed_log["has_security_keywords"] = self._has_security_keywords(message)
            
            processed.append(processed_log)
        
        return processed
    
    def _clean_message(self, message: str) -> str:
        """Clean log message"""
        # Remove extra whitespace
        message = " ".join(message.split())
        
        # Remove very long hex strings (likely data blobs)
        import re
        message = re.sub(r'[0-9a-f]{32,}', '[HEX_DATA]', message, flags=re.IGNORECASE)
        
        return message
    
    def _has_error_keywords(self, message: str) -> bool:
        """Check if message contains error-related keywords"""
        error_keywords = [
            "error", "failed", "failure", "exception", "crash",
            "fatal", "critical", "panic", "abort"
        ]
        message_lower = message.lower()
        return any(kw in message_lower for kw in error_keywords)
    
    def _has_security_keywords(self, message: str) -> bool:
        """Check if message contains security-related keywords"""
        security_keywords = [
            "attack", "malicious", "intrusion", "unauthorized", "denied",
            "blocked", "violation", "exploit", "injection", "breach"
        ]
        message_lower = message.lower()
        return any(kw in message_lower for kw in security_keywords)
    
    def split_train_test(self, logs: List[Dict], 
                         train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """Split data into training and test sets"""
        random.shuffle(logs)
        split_idx = int(len(logs) * train_ratio)
        return logs[:split_idx], logs[split_idx:]
    
    def stratified_split(self, logs: List[Dict], 
                         stratify_key: str = "severity",
                         train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """Stratified split maintaining class distribution"""
        # Group by stratification key
        groups = defaultdict(list)
        for log in logs:
            key = log.get(stratify_key, "unknown")
            groups[key].append(log)
        
        train_set = []
        test_set = []
        
        for key, group_logs in groups.items():
            random.shuffle(group_logs)
            split_idx = int(len(group_logs) * train_ratio)
            train_set.extend(group_logs[:split_idx])
            test_set.extend(group_logs[split_idx:])
        
        random.shuffle(train_set)
        random.shuffle(test_set)
        
        return train_set, test_set


class ModelEvaluator:
    """
    Evaluates trained ML models.
    
    Metrics:
    - Accuracy, Precision, Recall, F1
    - Confusion matrix
    - Per-class performance
    """
    
    def __init__(self):
        pass
    
    def calculate_metrics(self, predictions: List[str], 
                         ground_truth: List[str]) -> Dict[str, Any]:
        """Calculate classification metrics"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        classes = list(set(ground_truth))
        
        # Confusion matrix
        confusion = defaultdict(lambda: defaultdict(int))
        for pred, true in zip(predictions, ground_truth):
            confusion[true][pred] += 1
        
        # Per-class metrics
        per_class = {}
        for cls in classes:
            tp = confusion[cls][cls]
            fp = sum(confusion[other][cls] for other in classes if other != cls)
            fn = sum(confusion[cls][other] for other in classes if other != cls)
            tn = sum(
                confusion[other1][other2] 
                for other1 in classes if other1 != cls 
                for other2 in classes if other2 != cls
            )
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class[cls] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn
            }
        
        # Overall metrics
        correct = sum(1 for p, t in zip(predictions, ground_truth) if p == t)
        accuracy = correct / len(predictions) if predictions else 0
        
        # Macro average
        macro_precision = sum(m["precision"] for m in per_class.values()) / len(classes) if classes else 0
        macro_recall = sum(m["recall"] for m in per_class.values()) / len(classes) if classes else 0
        macro_f1 = sum(m["f1"] for m in per_class.values()) / len(classes) if classes else 0
        
        return {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "per_class": per_class,
            "confusion_matrix": dict(confusion),
            "total_samples": len(predictions)
        }
    
    def evaluate_anomaly_detector(self, detector: AnomalyDetector, 
                                  test_logs: List[Dict]) -> Dict[str, Any]:
        """Evaluate anomaly detector performance"""
        # Ground truth based on severity
        ground_truth = [
            1 if log.get("severity") in ["ERROR", "CRITICAL"] or 
                 log.get("has_error_keywords", False) else 0
            for log in test_logs
        ]
        
        # Get predictions (sync wrapper for evaluation)
        import asyncio
        results = asyncio.get_event_loop().run_until_complete(
            detector.detect(test_logs)
        )
        predictions = [1 if r.is_anomaly else 0 for r in results]
        
        # Calculate metrics
        tp = sum(1 for p, t in zip(predictions, ground_truth) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(predictions, ground_truth) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(predictions, ground_truth) if p == 0 and t == 1)
        tn = sum(1 for p, t in zip(predictions, ground_truth) if p == 0 and t == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "anomaly_rate": sum(predictions) / len(predictions) if predictions else 0
        }


class TrainingPipeline:
    """
    Main training pipeline that orchestrates all model training.
    """
    
    def __init__(self, 
                 logs_directory: Optional[str] = None,
                 models_directory: Optional[str] = None):
        # Use settings for default paths (absolute paths from project root)
        logs_dir = logs_directory or str(settings.logs_folder_resolved / "loghub")
        models_dir = models_directory or settings.models_dir
        
        self.data_loader = LogDataLoader(logs_dir)
        self.preprocessor = DataPreprocessor()
        self.evaluator = ModelEvaluator()
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.anomaly_detector = AnomalyDetector(
            model_path=str(self.models_dir / "anomaly_detector.pkl")
        )
        self.log_classifier = LogClassifier(
            model_path=str(self.models_dir / "log_classifier.pkl")
        )
        self.security_detector = SecurityThreatDetector(
            model_path=str(self.models_dir / "security_detector.pkl")
        )
        
        self.training_history: List[Dict] = []
        
    async def run_full_pipeline(self, 
                                max_logs_per_source: int = 2000,
                                train_ratio: float = 0.8) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            max_logs_per_source: Maximum logs to load per source
            train_ratio: Train/test split ratio
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting full training pipeline...")
        start_time = datetime.now()
        
        results = {
            "start_time": start_time.isoformat(),
            "stages": {},
            "models": {},
            "errors": []
        }
        
        # Stage 1: Load Data
        logger.info("Stage 1: Loading log data...")
        try:
            raw_logs = await self.data_loader.load_all_logs(max_logs_per_source)
            results["stages"]["data_loading"] = {
                "status": "success",
                "logs_loaded": len(raw_logs)
            }
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            results["stages"]["data_loading"] = {"status": "failed", "error": str(e)}
            results["errors"].append(f"Data loading: {e}")
            return results
        
        if len(raw_logs) < 100:
            logger.warning("Insufficient training data")
            results["stages"]["data_loading"]["warning"] = "Insufficient data"
            # Generate synthetic data for demo
            raw_logs = self._generate_synthetic_logs(1000)
            results["stages"]["data_loading"]["synthetic_logs"] = 1000
        
        # Stage 2: Preprocess Data
        logger.info("Stage 2: Preprocessing data...")
        try:
            processed_logs = self.preprocessor.preprocess(raw_logs)
            train_logs, test_logs = self.preprocessor.stratified_split(
                processed_logs, stratify_key="severity", train_ratio=train_ratio
            )
            results["stages"]["preprocessing"] = {
                "status": "success",
                "total_logs": len(processed_logs),
                "train_logs": len(train_logs),
                "test_logs": len(test_logs)
            }
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            results["stages"]["preprocessing"] = {"status": "failed", "error": str(e)}
            results["errors"].append(f"Preprocessing: {e}")
            return results
        
        # Stage 3: Train Anomaly Detector
        logger.info("Stage 3: Training anomaly detector...")
        try:
            anomaly_result = await self.anomaly_detector.train(train_logs)
            results["models"]["anomaly_detector"] = anomaly_result
        except Exception as e:
            logger.error(f"Anomaly detector training failed: {e}")
            results["models"]["anomaly_detector"] = {"status": "failed", "error": str(e)}
            results["errors"].append(f"Anomaly detector: {e}")
        
        # Stage 4: Train Log Classifier
        logger.info("Stage 4: Training log classifier...")
        try:
            classifier_result = await self.log_classifier.train(train_logs)
            results["models"]["log_classifier"] = classifier_result
        except Exception as e:
            logger.error(f"Log classifier training failed: {e}")
            results["models"]["log_classifier"] = {"status": "failed", "error": str(e)}
            results["errors"].append(f"Log classifier: {e}")
        
        # Stage 5: Train Security Detector
        logger.info("Stage 5: Training security threat detector...")
        try:
            # Separate normal and potential attack logs
            normal_logs = [l for l in train_logs if not l.get("has_security_keywords", False)]
            attack_logs = [l for l in train_logs if l.get("has_security_keywords", False)]
            
            security_result = await self.security_detector.train(
                normal_logs=normal_logs,
                attack_logs=attack_logs if attack_logs else None
            )
            results["models"]["security_detector"] = security_result
        except Exception as e:
            logger.error(f"Security detector training failed: {e}")
            results["models"]["security_detector"] = {"status": "failed", "error": str(e)}
            results["errors"].append(f"Security detector: {e}")
        
        # Stage 6: Evaluate Models
        logger.info("Stage 6: Evaluating models...")
        try:
            # Evaluate classifier
            classifications = await self.log_classifier.classify(test_logs)
            predicted_categories = [c.category for c in classifications]
            actual_categories = [
                "error" if l.get("has_error_keywords") else 
                "security" if l.get("has_security_keywords") else "normal"
                for l in test_logs
            ]
            
            classifier_metrics = self.evaluator.calculate_metrics(
                predicted_categories, actual_categories
            )
            results["stages"]["evaluation"] = {
                "status": "success",
                "classifier_metrics": classifier_metrics
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            results["stages"]["evaluation"] = {"status": "failed", "error": str(e)}
        
        # Finalize
        end_time = datetime.now()
        results["end_time"] = end_time.isoformat()
        results["duration_seconds"] = (end_time - start_time).total_seconds()
        results["success"] = len(results["errors"]) == 0
        
        # Record in history
        self.training_history.append({
            "timestamp": start_time.isoformat(),
            "duration": results["duration_seconds"],
            "logs_trained": len(train_logs),
            "success": results["success"]
        })
        
        logger.info(f"Training pipeline completed in {results['duration_seconds']:.2f}s")
        return results
    
    def _generate_synthetic_logs(self, count: int) -> List[Dict]:
        """Generate synthetic log data for training when real data is unavailable"""
        templates = [
            ("INFO", "User {user} logged in successfully from {ip}"),
            ("INFO", "Request processed in {time}ms"),
            ("WARNING", "High memory usage detected: {percent}%"),
            ("WARNING", "Slow query detected: {query}"),
            ("ERROR", "Connection to {service} failed: {error}"),
            ("ERROR", "Exception in {module}: {exception}"),
            ("CRITICAL", "Service {service} is down"),
            ("INFO", "Scheduled task {task} completed"),
            ("WARNING", "Failed login attempt from {ip}"),
            ("ERROR", "Authentication failed for user {user}"),
            ("INFO", "File {file} uploaded successfully"),
            ("WARNING", "Disk space low on {volume}"),
            ("ERROR", "Database query timeout: {query}"),
            ("INFO", "API request: {method} {endpoint}"),
            ("WARNING", "Rate limit exceeded for {ip}")
        ]
        
        logs = []
        for i in range(count):
            severity, template = random.choice(templates)
            
            # Generate realistic values
            message = template.format(
                user=f"user{random.randint(1, 100)}",
                ip=f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                time=random.randint(10, 5000),
                percent=random.randint(60, 99),
                query=f"SELECT * FROM table{random.randint(1, 10)}",
                service=random.choice(["database", "cache", "api", "auth"]),
                error=random.choice(["timeout", "refused", "reset"]),
                module=random.choice(["auth", "payment", "user", "order"]),
                exception=random.choice(["NullPointer", "OutOfMemory", "IO"]),
                task=random.choice(["backup", "cleanup", "report", "sync"]),
                file=f"file_{random.randint(1, 1000)}.txt",
                volume=random.choice(["/dev/sda1", "/dev/sdb1", "C:"]),
                method=random.choice(["GET", "POST", "PUT", "DELETE"]),
                endpoint=f"/api/v1/{random.choice(['users', 'orders', 'products'])}"
            )
            
            logs.append({
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "severity": severity,
                "source": f"synthetic_{i % 5}",
                "domain": "application"
            })
        
        return logs
    
    async def load_models(self) -> Dict[str, bool]:
        """Load all trained models from disk"""
        results = {}
        
        results["anomaly_detector"] = await self.anomaly_detector.load()
        results["log_classifier"] = await self.log_classifier.load()
        results["security_detector"] = await self.security_detector.load()
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline and model status"""
        return {
            "models": {
                "anomaly_detector": self.anomaly_detector.get_training_status(),
                "log_classifier": self.log_classifier.get_training_status(),
                "security_detector": self.security_detector.get_training_status()
            },
            "training_history": self.training_history[-10:],
            "models_directory": str(self.models_dir),
            "models_exist": {
                "anomaly_detector": (self.models_dir / "anomaly_detector.pkl").exists(),
                "log_classifier": (self.models_dir / "log_classifier.pkl").exists(),
                "security_detector": (self.models_dir / "security_detector.pkl").exists()
            }
        }


# Singleton instance for the application
_pipeline_instance: Optional[TrainingPipeline] = None


def get_training_pipeline() -> TrainingPipeline:
    """Get or create the training pipeline instance"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = TrainingPipeline()
    return _pipeline_instance


async def initialize_models():
    """Initialize models on application startup"""
    pipeline = get_training_pipeline()
    
    # Try to load existing models
    load_results = await pipeline.load_models()
    
    loaded_count = sum(1 for v in load_results.values() if v)
    
    if loaded_count < 3:
        logger.info("Some models not found, triggering training pipeline...")
        # Run training in background
        asyncio.create_task(pipeline.run_full_pipeline())
    else:
        logger.info("All models loaded successfully")
    
    return load_results
