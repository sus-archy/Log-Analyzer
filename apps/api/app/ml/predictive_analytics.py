"""
Predictive Analytics Service

This module implements predictive capabilities for log analysis:
1. Failure prediction using time-series forecasting
2. Trend detection and pattern recognition
3. Capacity planning predictions
4. Alert forecasting

Uses ARIMA-like models and exponential smoothing for time-series prediction.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of predictive analysis"""
    metric: str
    current_value: float
    predicted_value: float
    prediction_time: datetime
    confidence_interval: Tuple[float, float]
    trend: str  # 'increasing', 'decreasing', 'stable'
    risk_level: str  # 'critical', 'high', 'medium', 'low', 'none'
    explanation: str
    recommendations: List[str] = field(default_factory=list)


class ExponentialSmoothing:
    """
    Simple Exponential Smoothing for time-series forecasting.
    
    Suitable for data without trend or seasonality.
    """
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.level: Optional[float] = None
        self.fitted = False
        
    def fit(self, values: List[float]) -> "ExponentialSmoothing":
        """Fit the model to historical data"""
        if not values:
            return self
        
        self.level = values[0]
        for value in values[1:]:
            self.level = self.alpha * value + (1 - self.alpha) * self.level
        
        self.fitted = True
        return self
    
    def predict(self, steps: int = 1) -> List[float]:
        """Predict future values"""
        if not self.fitted or self.level is None:
            return [0.0] * steps
        return [self.level] * steps
    
    def update(self, value: float) -> float:
        """Update model with new observation"""
        if self.level is None:
            self.level = value
        else:
            self.level = self.alpha * value + (1 - self.alpha) * self.level
        self.fitted = True
        return self.level


class HoltLinearTrend:
    """
    Holt's Linear Trend Method (Double Exponential Smoothing).
    
    Captures both level and trend in the data.
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha  # Level smoothing
        self.beta = beta    # Trend smoothing
        self.level: float = 0.0
        self.trend: float = 0.0
        self.fitted = False
        
    def fit(self, values: List[float]) -> "HoltLinearTrend":
        """Fit the model to historical data"""
        if len(values) < 2:
            return self
        
        # Initialize level and trend
        self.level = values[0]
        self.trend = values[1] - values[0]
        
        # Fit through data
        for value in values[1:]:
            prev_level = self.level
            self.level = self.alpha * value + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
        
        self.fitted = True
        return self
    
    def predict(self, steps: int = 1) -> List[float]:
        """Predict future values"""
        if not self.fitted:
            return [0.0] * steps
        
        predictions = []
        for h in range(1, steps + 1):
            predictions.append(self.level + h * self.trend)
        return predictions
    
    def update(self, value: float) -> Tuple[float, float]:
        """Update model with new observation"""
        if not self.fitted:
            self.level = value
            self.trend = 0.0
        else:
            prev_level = self.level
            self.level = self.alpha * value + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
        
        self.fitted = True
        return self.level, self.trend


class SeasonalDecomposition:
    """
    Seasonal decomposition for detecting patterns.
    
    Decomposes time series into:
    - Trend component
    - Seasonal component
    - Residual component
    """
    
    def __init__(self, period: int = 24):  # Default: hourly seasonality in daily data
        self.period = period
        self.seasonal: Optional[np.ndarray] = None
        self.trend: Optional[np.ndarray] = None
        self.fitted = False
        
    def fit(self, values_list: List[float]) -> "SeasonalDecomposition":
        """Decompose the time series"""
        if len(values_list) < 2 * self.period:
            logger.warning("Insufficient data for seasonal decomposition")
            return self
        
        values = np.array(values_list)
        n = len(values)
        
        # Calculate trend using moving average
        half_period = self.period // 2
        self.trend = np.full(n, np.nan)
        
        for i in range(half_period, n - half_period):
            self.trend[i] = np.mean(values[i - half_period:i + half_period + 1])
        
        # Calculate seasonal component
        detrended = values - np.where(np.isnan(self.trend), values, self.trend)
        
        self.seasonal = np.zeros(self.period)
        for i in range(self.period):
            indices = np.arange(i, n, self.period)
            valid = ~np.isnan(detrended[indices])
            if valid.any():
                self.seasonal[i] = np.mean(detrended[indices][valid])
        
        # Normalize seasonal component
        self.seasonal = self.seasonal - np.mean(self.seasonal)
        
        self.fitted = True
        return self
    
    def get_seasonal_factor(self, position: int) -> float:
        """Get seasonal factor for a given position"""
        if self.seasonal is None:
            return 0.0
        return float(self.seasonal[position % self.period])
    
    def detect_anomalous_season(self, values: List[float], threshold: float = 2.0) -> List[int]:
        """Detect positions with anomalous seasonal behavior"""
        if not self.fitted or len(values) < self.period or self.seasonal is None:
            return []
        
        seasonal_std = float(np.std(self.seasonal))
        anomalies = []
        for i, value in enumerate(values):
            expected = self.get_seasonal_factor(i)
            if abs(value - expected) > threshold * seasonal_std:
                anomalies.append(i)
        
        return anomalies


class FailurePredictor:
    """
    Predicts system failures based on error rate patterns.
    
    Uses:
    - Error rate trending
    - Threshold breach prediction
    - Pattern-based failure detection
    """
    
    def __init__(self, error_threshold: float = 0.1, window_size: int = 60):
        self.error_threshold = error_threshold
        self.window_size = window_size
        
        self.error_history: deque = deque(maxlen=1000)
        self.model = HoltLinearTrend(alpha=0.3, beta=0.1)
        self.baseline_error_rate: float = 0.01
        self.baseline_std: float = 0.01
        self.trained = False
        
    def fit(self, error_rates: List[float]) -> "FailurePredictor":
        """Train on historical error rates"""
        if not error_rates:
            return self
        
        self.error_history.extend(error_rates)
        self.model.fit(error_rates)
        
        self.baseline_error_rate = float(np.mean(error_rates))
        self.baseline_std = float(np.std(error_rates)) + 1e-8
        
        self.trained = True
        logger.info(f"Failure predictor trained: baseline={self.baseline_error_rate:.4f}")
        return self
    
    def update(self, error_rate: float) -> None:
        """Update model with new error rate observation"""
        self.error_history.append(error_rate)
        self.model.update(error_rate)
        
        # Update baseline with exponential moving average
        alpha = 0.01
        self.baseline_error_rate = alpha * error_rate + (1 - alpha) * self.baseline_error_rate
    
    def predict(self, horizon_minutes: int = 60) -> Dict[str, Any]:
        """Predict failure probability over the given horizon"""
        if not self.trained or len(self.error_history) < 10:
            return {
                "failure_probability": 0.5,
                "time_to_failure": None,
                "confidence": 0.0,
                "trend": "unknown"
            }
        
        # Get predictions
        steps = horizon_minutes
        predictions = self.model.predict(steps)
        
        # Calculate when threshold will be breached
        time_to_failure = None
        for i, pred in enumerate(predictions):
            if pred > self.error_threshold:
                time_to_failure = i + 1
                break
        
        # Calculate failure probability
        current_rate = self.error_history[-1] if self.error_history else 0
        z_score = (current_rate - self.baseline_error_rate) / self.baseline_std
        
        # Sigmoid-based probability
        failure_prob = 1 / (1 + np.exp(-z_score))
        
        # Adjust based on trend
        if self.model.trend and self.model.trend > 0:
            failure_prob = min(failure_prob * 1.2, 1.0)
        
        # Determine trend
        trend = "stable"
        if self.model.trend:
            if self.model.trend > 0.001:
                trend = "increasing"
            elif self.model.trend < -0.001:
                trend = "decreasing"
        
        return {
            "failure_probability": float(failure_prob),
            "time_to_failure": time_to_failure,
            "predicted_rate": predictions[-1] if predictions else current_rate,
            "current_rate": current_rate,
            "trend": trend,
            "trend_value": float(self.model.trend) if self.model.trend else 0,
            "confidence": 0.8 if len(self.error_history) > 100 else 0.5
        }


class CapacityPredictor:
    """
    Predicts capacity exhaustion for various resources.
    
    Tracks metrics like:
    - Log volume growth
    - Storage utilization
    - Request rate growth
    """
    
    def __init__(self):
        self.metric_models: Dict[str, HoltLinearTrend] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.capacity_limits: Dict[str, float] = {
            "log_volume": 1000000,  # 1M logs
            "storage_gb": 100,      # 100GB
            "requests_per_minute": 10000
        }
        
    def set_capacity_limit(self, metric: str, limit: float) -> None:
        """Set capacity limit for a metric"""
        self.capacity_limits[metric] = limit
    
    def update(self, metric: str, value: float) -> None:
        """Update metric value"""
        self.metric_history[metric].append(value)
        
        if metric not in self.metric_models:
            self.metric_models[metric] = HoltLinearTrend(alpha=0.3, beta=0.1)
        
        self.metric_models[metric].update(value)
    
    def fit(self, metric: str, values: List[float]) -> None:
        """Train on historical values for a metric"""
        self.metric_history[metric].extend(values)
        
        if metric not in self.metric_models:
            self.metric_models[metric] = HoltLinearTrend(alpha=0.3, beta=0.1)
        
        self.metric_models[metric].fit(values)
    
    def predict_exhaustion(self, metric: str, 
                           horizon_hours: int = 24) -> Dict[str, Any]:
        """Predict when capacity will be exhausted"""
        if metric not in self.metric_models or not self.metric_models[metric].fitted:
            return {
                "metric": metric,
                "status": "no_data",
                "hours_to_exhaustion": None
            }
        
        model = self.metric_models[metric]
        limit = self.capacity_limits.get(metric, float('inf'))
        
        # Predict future values
        predictions = model.predict(horizon_hours)
        current = self.metric_history[metric][-1] if self.metric_history[metric] else 0
        
        # Find exhaustion point
        hours_to_exhaustion = None
        for i, pred in enumerate(predictions):
            if pred >= limit:
                hours_to_exhaustion = i + 1
                break
        
        # Calculate utilization
        utilization = current / limit if limit > 0 else 0
        predicted_utilization = predictions[-1] / limit if limit > 0 and predictions else utilization
        
        return {
            "metric": metric,
            "current_value": current,
            "current_utilization": utilization,
            "predicted_value": predictions[-1] if predictions else current,
            "predicted_utilization": predicted_utilization,
            "capacity_limit": limit,
            "hours_to_exhaustion": hours_to_exhaustion,
            "trend": "increasing" if model.trend and model.trend > 0 else "stable",
            "growth_rate": float(model.trend) if model.trend else 0
        }


class AlertPredictor:
    """
    Predicts future alerts based on current patterns.
    
    Uses historical alert data to forecast:
    - Expected alert volume
    - Types of alerts likely to occur
    - Optimal alerting thresholds
    """
    
    def __init__(self, lookback_hours: int = 24):
        self.lookback_hours = lookback_hours
        self.alert_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.hourly_model = SeasonalDecomposition(period=24)
        self.alert_counts: deque = deque(maxlen=168)  # 1 week of hourly data
        
    def record_alert(self, alert_type: str, timestamp: Optional[datetime] = None) -> None:
        """Record an alert occurrence"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.alert_patterns[alert_type].append(timestamp)
        
        # Keep only recent history
        cutoff = datetime.now() - timedelta(hours=self.lookback_hours * 7)
        self.alert_patterns[alert_type] = [
            t for t in self.alert_patterns[alert_type] if t > cutoff
        ]
    
    def fit(self, alerts: List[Dict[str, Any]]) -> "AlertPredictor":
        """Train on historical alert data"""
        for alert in alerts:
            alert_type = alert.get("type", "unknown")
            timestamp = alert.get("timestamp")
            
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except:
                    timestamp = datetime.now()
            
            self.record_alert(alert_type, timestamp)
        
        # Build hourly counts for seasonal decomposition
        hourly_counts = self._aggregate_hourly()
        if len(hourly_counts) >= 48:
            self.hourly_model.fit(hourly_counts)
        
        logger.info(f"Alert predictor trained on {len(alerts)} alerts")
        return self
    
    def _aggregate_hourly(self) -> List[float]:
        """Aggregate alerts into hourly counts"""
        all_timestamps = []
        for timestamps in self.alert_patterns.values():
            all_timestamps.extend(timestamps)
        
        if not all_timestamps:
            return []
        
        # Group by hour
        hourly = defaultdict(int)
        for ts in all_timestamps:
            hour_key = ts.replace(minute=0, second=0, microsecond=0)
            hourly[hour_key] += 1
        
        # Sort and return counts
        sorted_hours = sorted(hourly.keys())
        return [hourly[h] for h in sorted_hours]
    
    def predict_alerts(self, horizon_hours: int = 24) -> Dict[str, Any]:
        """Predict alert volume for the next period"""
        # Get seasonal factors
        current_hour = datetime.now().hour
        
        expected_by_hour = []
        for h in range(horizon_hours):
            hour = (current_hour + h) % 24
            seasonal = self.hourly_model.get_seasonal_factor(hour)
            expected_by_hour.append(seasonal)
        
        # Calculate per-type predictions
        type_predictions = {}
        for alert_type, timestamps in self.alert_patterns.items():
            # Recent rate
            recent = [t for t in timestamps if t > datetime.now() - timedelta(hours=24)]
            hourly_rate = len(recent) / 24 if recent else 0
            
            type_predictions[alert_type] = {
                "expected_count": hourly_rate * horizon_hours,
                "recent_rate": hourly_rate
            }
        
        total_expected = sum(p["expected_count"] for p in type_predictions.values())
        
        return {
            "horizon_hours": horizon_hours,
            "total_expected_alerts": total_expected,
            "by_type": type_predictions,
            "peak_hours": self._find_peak_hours(),
            "hourly_forecast": expected_by_hour
        }
    
    def _find_peak_hours(self) -> List[int]:
        """Find hours with highest alert activity"""
        if self.hourly_model.seasonal is None:
            return []
        
        # Get top 3 hours
        indices = np.argsort(self.hourly_model.seasonal)[::-1][:3]
        return indices.tolist()


class PredictiveAnalytics:
    """
    Main predictive analytics service combining all predictors.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.failure_predictor = FailurePredictor()
        self.capacity_predictor = CapacityPredictor()
        self.alert_predictor = AlertPredictor()
        
        self.model_path = Path(model_path) if model_path else Path("data/models/predictive.pkl")
        self.is_trained = False
        
    async def train(self, 
                    error_rates: Optional[List[float]] = None,
                    metrics: Optional[Dict[str, List[float]]] = None,
                    alerts: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Train all predictive models.
        """
        results = {"success": True, "models_trained": []}
        
        if error_rates:
            self.failure_predictor.fit(error_rates)
            results["models_trained"].append("failure_predictor")
        
        if metrics:
            for metric_name, values in metrics.items():
                self.capacity_predictor.fit(metric_name, values)
            results["models_trained"].append("capacity_predictor")
        
        if alerts:
            self.alert_predictor.fit(alerts)
            results["models_trained"].append("alert_predictor")
        
        self.is_trained = len(results["models_trained"]) > 0
        
        await self.save()
        
        return results
    
    async def predict(self, 
                      current_error_rate: Optional[float] = None,
                      current_metrics: Optional[Dict[str, float]] = None,
                      horizon_hours: int = 24) -> Dict[str, Any]:
        """
        Get predictions across all predictors.
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "horizon_hours": horizon_hours
        }
        
        # Update and get failure prediction
        if current_error_rate is not None:
            self.failure_predictor.update(current_error_rate)
            results["failure_prediction"] = self.failure_predictor.predict(horizon_hours * 60)
        
        # Update and get capacity predictions
        if current_metrics:
            capacity_predictions = {}
            for metric, value in current_metrics.items():
                self.capacity_predictor.update(metric, value)
                capacity_predictions[metric] = self.capacity_predictor.predict_exhaustion(
                    metric, horizon_hours
                )
            results["capacity_predictions"] = capacity_predictions
        
        # Get alert predictions
        results["alert_predictions"] = self.alert_predictor.predict_alerts(horizon_hours)
        
        # Generate overall risk assessment
        results["risk_assessment"] = self._assess_overall_risk(results)
        
        return results
    
    def _assess_overall_risk(self, predictions: Dict) -> Dict[str, Any]:
        """Assess overall system risk based on all predictions"""
        risk_factors = []
        
        # Check failure prediction
        if "failure_prediction" in predictions:
            fp = predictions["failure_prediction"]
            if fp.get("failure_probability", 0) > 0.7:
                risk_factors.append({
                    "factor": "high_failure_probability",
                    "severity": "critical",
                    "value": fp["failure_probability"]
                })
            elif fp.get("trend") == "increasing":
                risk_factors.append({
                    "factor": "increasing_error_trend",
                    "severity": "medium",
                    "value": fp.get("trend_value", 0)
                })
        
        # Check capacity predictions
        if "capacity_predictions" in predictions:
            for metric, cp in predictions["capacity_predictions"].items():
                if cp.get("hours_to_exhaustion") and cp["hours_to_exhaustion"] < 24:
                    risk_factors.append({
                        "factor": f"{metric}_exhaustion",
                        "severity": "high",
                        "hours_remaining": cp["hours_to_exhaustion"]
                    })
                elif cp.get("predicted_utilization", 0) > 0.8:
                    risk_factors.append({
                        "factor": f"{metric}_high_utilization",
                        "severity": "medium",
                        "utilization": cp["predicted_utilization"]
                    })
        
        # Calculate overall risk score
        severity_scores = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.2}
        risk_score = 0.0
        
        for factor in risk_factors:
            risk_score = max(risk_score, severity_scores.get(factor["severity"], 0))
        
        # Determine risk level
        if risk_score >= 0.9:
            risk_level = "critical"
        elif risk_score >= 0.6:
            risk_level = "high"
        elif risk_score >= 0.3:
            risk_level = "medium"
        elif risk_score > 0:
            risk_level = "low"
        else:
            risk_level = "none"
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendations": self._generate_recommendations(risk_factors)
        }
    
    def _generate_recommendations(self, risk_factors: List[Dict]) -> List[str]:
        """Generate recommendations based on risk factors"""
        recommendations = []
        
        for factor in risk_factors:
            if "failure_probability" in factor["factor"]:
                recommendations.append("Review recent error logs for root cause")
                recommendations.append("Consider scaling up resources")
            elif "exhaustion" in factor["factor"]:
                metric = factor["factor"].replace("_exhaustion", "")
                recommendations.append(f"Increase {metric} capacity")
                recommendations.append(f"Implement {metric} cleanup or archival")
            elif "utilization" in factor["factor"]:
                recommendations.append("Monitor resource utilization closely")
                recommendations.append("Plan for capacity expansion")
            elif "error_trend" in factor["factor"]:
                recommendations.append("Investigate cause of increasing errors")
                recommendations.append("Review recent deployments")
        
        return list(set(recommendations))[:5]
    
    async def save(self) -> None:
        """Save models to disk"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "failure_predictor": {
                "error_history": list(self.failure_predictor.error_history),
                "baseline_rate": self.failure_predictor.baseline_error_rate,
                "baseline_std": self.failure_predictor.baseline_std,
                "model_level": self.failure_predictor.model.level,
                "model_trend": self.failure_predictor.model.trend
            },
            "capacity_predictor": {
                "limits": self.capacity_predictor.capacity_limits,
                "history": {k: list(v) for k, v in self.capacity_predictor.metric_history.items()}
            },
            "alert_predictor": {
                "patterns": {k: [t.isoformat() for t in v] 
                            for k, v in self.alert_predictor.alert_patterns.items()}
            }
        }
        
        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)
    
    async def load(self) -> bool:
        """Load models from disk"""
        if not self.model_path.exists():
            return False
        
        try:
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
            
            # Restore failure predictor
            fp_data = model_data["failure_predictor"]
            self.failure_predictor.error_history = deque(fp_data["error_history"], maxlen=1000)
            self.failure_predictor.baseline_error_rate = fp_data["baseline_rate"]
            self.failure_predictor.baseline_std = fp_data["baseline_std"]
            self.failure_predictor.model.level = fp_data["model_level"]
            self.failure_predictor.model.trend = fp_data["model_trend"]
            self.failure_predictor.trained = True
            
            # Restore capacity predictor
            cp_data = model_data["capacity_predictor"]
            self.capacity_predictor.capacity_limits = cp_data["limits"]
            for k, v in cp_data["history"].items():
                self.capacity_predictor.fit(k, v)
            
            # Restore alert predictor
            ap_data = model_data["alert_predictor"]
            for alert_type, timestamps in ap_data["patterns"].items():
                for ts_str in timestamps:
                    self.alert_predictor.record_alert(
                        alert_type, 
                        datetime.fromisoformat(ts_str)
                    )
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load predictive models: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of predictive models"""
        return {
            "is_trained": self.is_trained,
            "failure_predictor": {
                "trained": self.failure_predictor.trained,
                "history_size": len(self.failure_predictor.error_history)
            },
            "capacity_predictor": {
                "metrics_tracked": list(self.capacity_predictor.metric_models.keys()),
                "limits_configured": self.capacity_predictor.capacity_limits
            },
            "alert_predictor": {
                "alert_types": list(self.alert_predictor.alert_patterns.keys()),
                "total_alerts": sum(
                    len(v) for v in self.alert_predictor.alert_patterns.values()
                )
            }
        }
