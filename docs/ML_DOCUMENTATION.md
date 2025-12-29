# Machine Learning Module Documentation

## Overview

LogMind AI includes a comprehensive machine learning module that provides **real** trained models for:

1. **Anomaly Detection** - Isolation Forest + Statistical methods
2. **Log Classification** - TF-IDF + Naive Bayes + SVM
3. **Security Threat Detection** - Attack pattern recognition
4. **Predictive Analytics** - Time-series forecasting

All models are trained on **actual log data** from the `Logs/loghub` directory.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ML Module Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐   │
│  │   Log Data      │────▶│ Training Pipeline│────▶│   Trained Models   │   │
│  │  (Logs/loghub)  │     │                  │     │  (data/models/)    │   │
│  └─────────────────┘     └──────────────────┘     └────────────────────┘   │
│                                                              │              │
│                                                              ▼              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        API Endpoints                                 │   │
│  ├──────────────┬──────────────┬─────────────────┬────────────────────┤   │
│  │ /ml/anomaly  │ /ml/classify │ /ml/security    │ /ml/predict        │   │
│  └──────────────┴──────────────┴─────────────────┴────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## ML Models Detail

### 1. Anomaly Detector

**Algorithm**: Custom Isolation Forest + Statistical Analysis

```python
# Core Components:
- IsolationForestDetector: Multivariate anomaly detection
- StatisticalDetector: Z-score, Modified Z-score, IQR methods
- TemporalAnomalyDetector: Time-based pattern analysis
```

**How it works:**
1. Builds isolation trees that randomly partition the feature space
2. Anomalies are isolated quickly (short path length)
3. Normal points require many partitions to isolate
4. Combines with statistical methods for robust detection

**Features extracted:**
- Log frequency
- Message length
- Severity distribution
- Template distribution
- Temporal patterns

### 2. Log Classifier

**Algorithm**: TF-IDF + Multinomial Naive Bayes + Linear SVM

```python
# Core Components:
- TFIDFVectorizer: Custom implementation for log text
- NaiveBayesClassifier: For category and domain classification
- SVMClassifier: For severity classification with SGD
```

**Classifications:**
- **Category**: error, warning, security, performance, normal
- **Domain**: web_server, database, authentication, system, network, distributed
- **Severity**: critical, high, medium, low, info

**Text Processing:**
- Log-specific tokenization (preserves IP addresses, timestamps, paths)
- N-gram extraction (unigrams and bigrams)
- TF-IDF weighting with L2 normalization

### 3. Security Threat Detector

**Algorithm**: Markov Chain Sequence Analysis + Pattern Detection

```python
# Core Components:
- AttackSequenceDetector: Markov chain for attack patterns
- BruteForceDetector: Statistical brute-force detection
- InjectionDetector: Pattern and entropy-based injection detection
- ReconnaissanceDetector: Scanning/enumeration detection
```

**Threat Types Detected:**
- Brute force attacks
- SQL injection
- Command injection
- XSS attacks
- Path traversal
- Port scanning
- Directory enumeration
- User enumeration

### 4. Predictive Analytics

**Algorithm**: Holt-Winters Exponential Smoothing + Seasonal Decomposition

```python
# Core Components:
- ExponentialSmoothing: For stable metrics
- HoltLinearTrend: For trending metrics
- SeasonalDecomposition: For cyclical patterns
- FailurePredictor: Predicts system failures
- CapacityPredictor: Predicts resource exhaustion
- AlertPredictor: Forecasts alert volume
```

## API Endpoints

### Training

```http
POST /ml/train
{
    "max_logs_per_source": 2000,
    "train_ratio": 0.8,
    "force_retrain": false
}

GET /ml/train/status
```

### Anomaly Detection

```http
POST /ml/anomaly/detect
{
    "logs": [
        {"message": "Connection failed to database", "severity": "ERROR"}
    ]
}

Response:
{
    "is_anomaly": true,
    "anomaly_score": 0.85,
    "anomaly_type": "severity",
    "confidence": 0.9,
    "explanation": "This log has high severity (ERROR) which deviates from normal patterns."
}
```

### Classification

```http
POST /ml/classify
{
    "logs": [
        {"message": "Failed login attempt from 192.168.1.100"}
    ]
}

Response:
{
    "category": "security",
    "domain": "authentication",
    "severity": "high",
    "confidence": 0.87,
    "probabilities": {"security": 0.87, "error": 0.08, "normal": 0.05}
}
```

### Security Detection

```http
POST /ml/security/detect
{
    "logs": [
        {"message": "SELECT * FROM users WHERE id=1 OR 1=1--"}
    ]
}

Response:
{
    "is_threat": true,
    "threat_score": 0.95,
    "threat_type": "sql_injection",
    "severity": "critical",
    "recommended_actions": [
        "Sanitize and validate all user inputs",
        "Use parameterized queries"
    ]
}
```

### Predictions

```http
POST /ml/predict
{
    "current_error_rate": 0.05,
    "current_metrics": {"log_volume": 50000},
    "horizon_hours": 24
}

Response:
{
    "failure_prediction": {
        "failure_probability": 0.3,
        "time_to_failure": null,
        "trend": "stable"
    },
    "capacity_predictions": {...},
    "risk_assessment": {
        "risk_score": 0.2,
        "risk_level": "low"
    }
}
```

### Full Analysis

```http
POST /ml/analyze
{
    "logs": [...]
}

# Returns combined anomaly, classification, and security analysis
```

## Training the Models

### Automatic Training

Models are trained automatically when:
1. API starts and no models exist
2. Training is triggered via `/ml/train` endpoint

### Manual Training

```bash
cd /home/bug/Desktop/Log_Analyzer
python scripts/train_models.py
```

### Training Data Requirements

- Minimum 100 log entries for anomaly detection
- Minimum 50 log entries for classification
- More data = better accuracy

### Model Persistence

Models are saved to `data/models/`:
- `anomaly_detector.pkl` - Isolation Forest + Statistical baselines
- `log_classifier.pkl` - TF-IDF vocabulary + Classifier weights
- `security_detector.pkl` - Attack pattern models
- `predictive.pkl` - Time-series models

## Performance Characteristics

| Model | Training Time (1000 logs) | Inference Time (single) | Inference Time (batch 100) |
|-------|---------------------------|-------------------------|---------------------------|
| Anomaly | ~2s | ~1ms | ~10ms |
| Classifier | ~3s | ~0.5ms | ~5ms |
| Security | ~1s | ~2ms | ~20ms |
| Predictive | ~0.5s | ~0.1ms | ~1ms |

## Accuracy Notes

The accuracy depends heavily on:
1. **Amount of training data** - More logs = better models
2. **Data quality** - Clean, labeled data improves accuracy
3. **Domain coverage** - Training on diverse log types improves generalization

Expected accuracy ranges (with sufficient training data):
- Anomaly detection: 70-90% (depends on anomaly definition)
- Classification: 80-95% (for well-defined categories)
- Security detection: 85-95% (for known attack patterns)

## Future Improvements

1. **Deep Learning**: Add LSTM/Transformer models for sequence analysis
2. **Online Learning**: Continuous model updates as new logs arrive
3. **Transfer Learning**: Pre-trained models for common log formats
4. **Ensemble Methods**: Combine multiple models for better accuracy
5. **AutoML**: Automatic hyperparameter tuning
