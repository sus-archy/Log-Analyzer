from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import json
import asyncio
import re

from ..ml import (
    get_training_pipeline,
    AnomalyDetector,
    LogClassifier,
    SecurityThreatDetector,
    PredictiveAnalytics
)
from ..llm.ollama_client import get_ollama_client, OllamaError
from ..storage.db import db_connection
from ..storage.logs_repo import LogsRepo
from ..core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ml", tags=["Machine Learning"])


# ============== Request/Response Models ==============

class LogEntry(BaseModel):
    """Single log entry for analysis"""
    message: str
    timestamp: Optional[str] = None
    severity: Optional[str] = "INFO"
    source: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Failed login attempt from IP 192.168.1.100",
                "timestamp": "2024-01-15T10:30:00Z",
                "severity": "WARNING",
                "source": "auth_service"
            }
        }


class BatchLogsRequest(BaseModel):
    """Batch of logs for analysis"""
    logs: List[LogEntry]


class TrainingRequest(BaseModel):
    """Request to trigger model training"""
    max_logs_per_source: int = Field(default=2000, ge=100, le=10000)
    train_ratio: float = Field(default=0.8, ge=0.5, le=0.95)
    force_retrain: bool = False


class AnomalyResponse(BaseModel):
    """Response from anomaly detection"""
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    confidence: float
    explanation: str
    contributing_factors: List[Dict[str, Any]] = []


class ClassificationResponse(BaseModel):
    """Response from log classification"""
    category: str
    domain: str
    severity: str
    confidence: float
    probabilities: Dict[str, float]
    extracted_entities: Dict[str, List[str]] = {}


class ThreatResponse(BaseModel):
    """Response from security threat detection"""
    is_threat: bool
    threat_score: float
    threat_type: str
    severity: str
    confidence: float
    attack_indicators: List[Dict[str, Any]] = []
    recommended_actions: List[str] = []


class PredictionRequest(BaseModel):
    """Request for predictive analysis"""
    current_error_rate: Optional[float] = None
    current_metrics: Optional[Dict[str, float]] = None
    horizon_hours: int = Field(default=24, ge=1, le=168)


# ============== Global Model Instances ==============

_anomaly_detector: Optional[AnomalyDetector] = None
_log_classifier: Optional[LogClassifier] = None
_security_detector: Optional[SecurityThreatDetector] = None
_predictive_analytics: Optional[PredictiveAnalytics] = None


async def get_anomaly_detector() -> AnomalyDetector:
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector()
        await _anomaly_detector.load()
    return _anomaly_detector


async def get_log_classifier() -> LogClassifier:
    global _log_classifier
    if _log_classifier is None:
        _log_classifier = LogClassifier()
        await _log_classifier.load()
    return _log_classifier


async def get_security_detector() -> SecurityThreatDetector:
    global _security_detector
    if _security_detector is None:
        _security_detector = SecurityThreatDetector()
        await _security_detector.load()
    return _security_detector


async def get_predictive_analytics() -> PredictiveAnalytics:
    global _predictive_analytics
    if _predictive_analytics is None:
        _predictive_analytics = PredictiveAnalytics()
        await _predictive_analytics.load()
    return _predictive_analytics


# ============== Training Endpoints ==============

@router.post("/train", response_model=Dict[str, Any])
async def train_models(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger training of all ML models.
    
    This runs the full training pipeline:
    1. Load logs from Logs/loghub directory
    2. Preprocess and label data
    3. Train anomaly detector, classifier, and security detector
    4. Evaluate and save models
    
    Training runs in the background and returns immediately.
    """
    pipeline = get_training_pipeline()
    
    # Check if models already exist and force_retrain is False
    status = pipeline.get_status()
    if not request.force_retrain and all(status["models_exist"].values()):
        return {
            "status": "skipped",
            "message": "Models already exist. Use force_retrain=true to retrain.",
            "existing_models": status["models_exist"]
        }
    
    # Run training in background
    async def run_training():
        try:
            result = await pipeline.run_full_pipeline(
                max_logs_per_source=request.max_logs_per_source,
                train_ratio=request.train_ratio
            )
            logger.info(f"Training completed: {result}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    background_tasks.add_task(run_training)
    
    return {
        "status": "started",
        "message": "Training pipeline started in background",
        "config": {
            "max_logs_per_source": request.max_logs_per_source,
            "train_ratio": request.train_ratio
        }
    }


@router.get("/train/status", response_model=Dict[str, Any])
async def get_training_status():
    """
    Get current status of ML models and training history.
    """
    pipeline = get_training_pipeline()
    return pipeline.get_status()


# ============== Anomaly Detection Endpoints ==============

@router.post("/anomaly/detect", response_model=List[AnomalyResponse])
async def detect_anomalies(request: BatchLogsRequest):
    """
    Detect anomalies in a batch of log entries.
    
    Uses trained Isolation Forest and statistical models to identify
    unusual log patterns.
    """
    detector = await get_anomaly_detector()
    
    logs = [log.model_dump() for log in request.logs]
    results = await detector.detect(logs)
    
    return [
        AnomalyResponse(
            is_anomaly=r.is_anomaly,
            anomaly_score=r.anomaly_score,
            anomaly_type=r.anomaly_type,
            confidence=r.confidence,
            explanation=r.explanation,
            contributing_factors=r.contributing_factors
        )
        for r in results
    ]


@router.post("/anomaly/detect/single", response_model=AnomalyResponse)
async def detect_single_anomaly(log: LogEntry):
    """
    Detect if a single log entry is anomalous.
    """
    detector = await get_anomaly_detector()
    
    results = await detector.detect([log.model_dump()])
    r = results[0]
    
    return AnomalyResponse(
        is_anomaly=r.is_anomaly,
        anomaly_score=r.anomaly_score,
        anomaly_type=r.anomaly_type,
        confidence=r.confidence,
        explanation=r.explanation,
        contributing_factors=r.contributing_factors
    )


# ============== Classification Endpoints ==============

@router.post("/classify", response_model=List[ClassificationResponse])
async def classify_logs(request: BatchLogsRequest):
    """
    Classify log entries by category, domain, and severity.
    
    Uses trained TF-IDF vectorizer with Naive Bayes and SVM classifiers.
    """
    classifier = await get_log_classifier()
    
    logs = [log.model_dump() for log in request.logs]
    results = await classifier.classify(logs)
    
    return [
        ClassificationResponse(
            category=r.category,
            domain=r.domain,
            severity=r.severity,
            confidence=r.confidence,
            probabilities=r.probabilities,
            extracted_entities=r.extracted_entities
        )
        for r in results
    ]


@router.post("/classify/single", response_model=ClassificationResponse)
async def classify_single_log(log: LogEntry):
    """
    Classify a single log entry.
    """
    classifier = await get_log_classifier()
    
    results = await classifier.classify([log.model_dump()])
    r = results[0]
    
    return ClassificationResponse(
        category=r.category,
        domain=r.domain,
        severity=r.severity,
        confidence=r.confidence,
        probabilities=r.probabilities,
        extracted_entities=r.extracted_entities
    )


# ============== Security Threat Detection Endpoints ==============

@router.post("/security/detect", response_model=List[ThreatResponse])
async def detect_security_threats(request: BatchLogsRequest):
    """
    Analyze logs for security threats.
    
    Uses ML-based detection for:
    - Brute force attacks
    - Injection attempts (SQL, command, XSS)
    - Reconnaissance/scanning
    - Attack patterns
    """
    detector = await get_security_detector()
    
    logs = [log.model_dump() for log in request.logs]
    results = await detector.detect(logs)
    
    return [
        ThreatResponse(
            is_threat=r.is_threat,
            threat_score=r.threat_score,
            threat_type=r.threat_type,
            severity=r.severity,
            confidence=r.confidence,
            attack_indicators=r.attack_indicators,
            recommended_actions=r.recommended_actions
        )
        for r in results
    ]


@router.post("/security/detect/single", response_model=ThreatResponse)
async def detect_single_threat(log: LogEntry):
    """
    Check a single log entry for security threats.
    """
    detector = await get_security_detector()
    
    results = await detector.detect([log.model_dump()])
    r = results[0]
    
    return ThreatResponse(
        is_threat=r.is_threat,
        threat_score=r.threat_score,
        threat_type=r.threat_type,
        severity=r.severity,
        confidence=r.confidence,
        attack_indicators=r.attack_indicators,
        recommended_actions=r.recommended_actions
    )


@router.get("/security/active-threats", response_model=Dict[str, Any])
async def get_active_threats():
    """
    Get currently tracked active threat sources.
    
    Returns sources with ongoing suspicious activity like brute force attempts.
    """
    detector = await get_security_detector()
    status = detector.get_training_status()
    
    return {
        "active_sources": status.get("active_sources", []),
        "detector_status": status["detectors"]
    }


# ============== Predictive Analytics Endpoints ==============

@router.post("/predict", response_model=Dict[str, Any])
async def get_predictions(request: PredictionRequest):
    """
    Get predictive analytics for system health.
    
    Predicts:
    - Failure probability based on error rate trends
    - Capacity exhaustion timing
    - Expected alert volume
    - Overall risk assessment
    """
    predictor = await get_predictive_analytics()
    
    return await predictor.predict(
        current_error_rate=request.current_error_rate,
        current_metrics=request.current_metrics,
        horizon_hours=request.horizon_hours
    )


@router.get("/predict/status", response_model=Dict[str, Any])
async def get_prediction_status():
    """
    Get status of predictive analytics models.
    """
    predictor = await get_predictive_analytics()
    return predictor.get_status()


# ============== Intelligent AI Chat System ==============

class MLChatRequest(BaseModel):
    """Request for intelligent ML-powered chat"""
    message: str = Field(..., min_length=1, description="Any question or log message")
    context: Optional[List[str]] = Field(default=None, description="Previous conversation messages")
    include_raw_analysis: bool = Field(default=False, description="Include raw ML analysis data")
    use_llm: bool = Field(default=True, description="Use LLM for enhanced responses (slower)")
    fast_mode: bool = Field(default=False, description="Skip LLM for instant responses")


class MLChatResponse(BaseModel):
    """Response from intelligent ML chat"""
    response: str
    analysis: Optional[Dict[str, Any]] = None
    model_status: Dict[str, bool]
    suggestions: List[str] = []


# System prompt for the LLM - shorter and more focused
LOG_EXPERT_SYSTEM_PROMPT = """You are LogMind AI, a log analysis expert. Be concise and helpful.
Given ML analysis results, briefly explain findings and give practical advice.
Keep responses under 200 words. Use markdown formatting."""

# Timeout for LLM calls (seconds)
LLM_TIMEOUT = 360.0  # 360 seconds for LLM response


async def _search_logs_in_database(search_terms: List[str], limit: int = 15, service_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search for logs in the database matching given search terms"""
    results = []
    
    try:
        async with db_connection() as db:
            logs_repo = LogsRepo(db)
            
            # Get time range for last 30 days
            now = datetime.utcnow()
            from_time = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
            to_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # If we have a service filter, search by service_name first
            if service_filter:
                cursor = await db.execute("""
                    SELECT id, body_raw, timestamp_utc, severity, service_name, template_hash
                    FROM logs_stream
                    WHERE service_name LIKE ?
                    ORDER BY timestamp_utc DESC
                    LIMIT ?
                """, (f"%{service_filter}%", limit))
                
                rows = await cursor.fetchall()
                for row in rows:
                    results.append({
                        'id': row['id'],
                        'message': row['body_raw'],
                        'timestamp': row['timestamp_utc'],
                        'severity': row['severity'],
                        'service': row['service_name'],
                    })
                
                # If we found service-specific logs, return those
                if results:
                    return results[:limit]
            
            # Search using SQL LIKE with each term in body_raw OR service_name
            for term in search_terms[:5]:  # Up to 5 search terms
                # Skip generic words that match too much
                if term.lower() in ['related', 'logs', 'log', 'show', 'find', 'give']:
                    continue
                    
                # Escape special SQL characters
                safe_term = term.replace('%', '\\%').replace('_', '\\_')
                
                # Search in both body_raw and service_name
                cursor = await db.execute("""
                    SELECT id, body_raw, timestamp_utc, severity, service_name, template_hash
                    FROM logs_stream
                    WHERE body_raw LIKE ? ESCAPE '\\' OR service_name LIKE ? ESCAPE '\\'
                    ORDER BY timestamp_utc DESC
                    LIMIT ?
                """, (f"%{safe_term}%", f"%{safe_term}%", limit * 2))
                
                rows = await cursor.fetchall()
                for row in rows:
                    # Avoid duplicates
                    if not any(r['id'] == row['id'] for r in results):
                        results.append({
                            'id': row['id'],
                            'message': row['body_raw'],
                            'timestamp': row['timestamp_utc'],
                            'severity': row['severity'],
                            'service': row['service_name'],
                        })
                
                if len(results) >= limit:
                    break
    
    except Exception as e:
        logger.warning(f"Failed to search logs: {e}")
    
    return results[:limit]


# Semantic mappings: user intent -> actual search terms to use
SEMANTIC_SEARCH_MAPPINGS = {
    # Authentication / Password issues
    'incorrect password': ['authentication failed', 'invalid password', 'wrong password', 'login failed', 'access denied'],
    'wrong password': ['authentication failed', 'invalid password', 'login failed', 'access denied'],
    'not correct pass': ['authentication failed', 'invalid password', 'login failed', 'access denied'],
    'bad password': ['authentication failed', 'invalid password', 'login failed'],
    'password fail': ['authentication failed', 'invalid password', 'login failed'],
    'auth fail': ['authentication failed', 'login failed', 'unauthorized', 'access denied'],
    'login fail': ['login failed', 'authentication failed', 'access denied'],
    'cant login': ['login failed', 'authentication failed', 'access denied'],
    'access denied': ['access denied', 'permission denied', 'unauthorized', 'forbidden'],
    
    # Connection issues
    'connection refused': ['connection refused', 'ECONNREFUSED', 'could not connect'],
    'timeout': ['timeout', 'timed out', 'connection timeout', 'request timeout'],
    'network error': ['network error', 'connection failed', 'unreachable', 'host down'],
    
    # Errors
    'out of memory': ['out of memory', 'OOM', 'memory allocation', 'cannot allocate'],
    'disk full': ['disk full', 'no space left', 'disk space', 'ENOSPC'],
    'file not found': ['file not found', 'ENOENT', 'no such file', 'does not exist'],
    
    # Security
    'sql injection': ['SQL injection', 'SQLi', 'SELECT FROM', 'UNION SELECT', '1=1'],
    'xss': ['XSS', 'cross-site scripting', '<script>', 'javascript:'],
    'brute force': ['brute force', 'multiple failed', 'too many attempts', 'rate limit'],
}

# Service name mappings are no longer hardcoded - we search dynamically
# This dictionary is kept for common abbreviations/aliases only
SERVICE_ALIASES = {
    'health app': 'health',
    'access log': 'access',
    'open ssh': 'ssh',
    'open stack': 'openstack',
}


def _detect_service_filter(user_message: str) -> Optional[str]:
    """Detect if user is asking for logs from a specific service/source - searches dynamically"""
    user_lower = user_message.lower()
    
    # Check for common aliases first
    for phrase, service_filter in SERVICE_ALIASES.items():
        if phrase in user_lower:
            return service_filter
    
    # Extract potential service names from the message
    # Look for patterns like "logs from X", "X logs", "related to X", "from X"
    service_patterns = [
        r'logs?\s+(?:from|of|for|related to)\s+(\w+)',
        r'(\w+)\s+logs?',
        r'related to\s+(\w+)',
        r'from\s+(\w+)',
        r'show me\s+(\w+)',
    ]
    
    # Common words to ignore (not service names)
    ignore_words = {'the', 'all', 'any', 'some', 'error', 'warning', 'failed', 'logs', 'log',
                    'show', 'find', 'get', 'give', 'me', 'please', 'can', 'you', 'with', 
                    'recent', 'latest', 'last', 'first', 'more', 'less', 'many', 'few'}
    
    for pattern in service_patterns:
        matches = re.findall(pattern, user_lower)
        for match in matches:
            # Clean the match
            candidate = match.strip()
            if len(candidate) >= 3 and candidate not in ignore_words:
                return candidate
    
    return None


def _extract_search_terms(user_message: str) -> List[str]:
    """Extract search terms from user's natural language query with semantic understanding"""
    user_lower = user_message.lower().strip()
    terms = []
    
    # ===== Check for semantic mappings first =====
    # This handles cases like "show me incorrect password" -> search for actual auth failure messages
    for phrase, search_terms in SEMANTIC_SEARCH_MAPPINGS.items():
        if phrase in user_lower:
            terms.extend(search_terms)
            break  # Use first matching semantic group
    
    # If we found semantic terms, use those
    if terms:
        return list(set(terms))[:5]
    
    # ===== Otherwise extract from patterns =====
    search_patterns = [
        r'where (?:is|are|can i find)(?: the)? (.+?)(?:\?|$)',
        r'show me (.+?)(?:\?|$)',
        r'find (.+?)(?:\?|$)',
        r'search for (.+?)(?:\?|$)',
        r'look(?:ing)? for (.+?)(?:\?|$)',
        r'logs? (?:with|containing|about|related to) (.+?)(?:\?|$)',
    ]
    
    # Extract from patterns
    for pattern in search_patterns:
        matches = re.findall(pattern, user_lower, re.IGNORECASE)
        for match in matches:
            # Clean and split the match
            words = match.strip().split()
            # Filter out common stop words and short words
            stop_words = ['the', 'and', 'for', 'are', 'was', 'logs', 'log', 'with', 'not', 'that', 'this', 'any', 'all']
            terms.extend([w for w in words if len(w) > 2 and w not in stop_words])
    
    # Also extract specific keywords commonly searched
    keywords = ['failed', 'error', 'login', 'password', 'authentication', 'timeout', 
                'connection', 'denied', 'unauthorized', 'exception', 'crash', 'warning',
                'success', 'blocked', 'invalid', 'expired', 'refused', 'forbidden',
                'critical', 'fatal', 'panic', 'alert', 'emergency']
    
    for kw in keywords:
        if kw in user_lower and kw not in terms:
            terms.append(kw)
    
    # If still no terms, extract nouns from the message (simple heuristic)
    if not terms:
        words = re.findall(r'\b[a-z]{4,}\b', user_lower)
        stop_words = ['show', 'find', 'give', 'look', 'search', 'logs', 'where', 'what', 'when', 'correct', 'incorrect']
        terms = [w for w in words if w not in stop_words][:3]
    
    return list(set(terms))[:5]  # Return unique terms, max 5


def _is_log_search_query(user_message: str) -> bool:
    """Detect if user is asking to search/find logs"""
    user_lower = user_message.lower()
    search_indicators = [
        'where is', 'where are', 'show me', 'find', 'search', 'look for', 'looking for',
        'can you find', 'can i see', 'give me', 'list', 'logs with', 'logs containing',
        'in the logs', 'from the logs', 'in my logs', 'examples of'
    ]
    return any(indicator in user_lower for indicator in search_indicators)


def _format_log_search_results(user_message: str, search_terms: List[str], logs: List[Dict], a: Dict, c: Dict, s: Dict) -> str:
    """Format log search results into a helpful response"""
    
    # Build the response
    terms_str = ', '.join(f'`{t}`' for t in search_terms)
    
    response_parts = [
        f"## 🔍 Log Search Results\n",
        f"**Searched for:** {terms_str}\n",
        f"**Found:** {len(logs)} matching log entries\n",
    ]
    
    if logs:
        # Determine how many logs to show based on total found
        show_count = min(len(logs), 15)  # Show up to 15 logs
        response_parts.append(f"\n### 📋 Showing {show_count} of {len(logs)} Matching Logs:\n")
        
        for i, log in enumerate(logs[:show_count], 1):
            timestamp = log.get('timestamp', 'N/A')
            severity = log.get('severity', 'INFO')
            service = log.get('service', 'unknown')
            message = log.get('message', '')
            
            # Truncate long messages
            if len(message) > 200:
                message = message[:200] + "..."
            
            # Convert numeric severity to string labels
            severity_map = {
                '0': 'DEBUG', '1': 'INFO', '2': 'WARNING', '3': 'ERROR', '4': 'CRITICAL',
                0: 'DEBUG', 1: 'INFO', 2: 'WARNING', 3: 'ERROR', 4: 'CRITICAL',
            }
            severity_str = severity_map.get(severity, str(severity).upper() if severity else 'INFO')
            
            # Color-code severity
            severity_icon = {
                'ERROR': '🔴',
                'WARNING': '🟡', 
                'WARN': '🟡',
                'INFO': '🔵',
                'DEBUG': '⚪',
                'CRITICAL': '🟣',
                'FATAL': '🟣',
            }.get(severity_str, '⚪')
            
            response_parts.append(f"**{i}.** {severity_icon} `{severity_str}`\n")
            response_parts.append(f"   - **Time:** {timestamp}\n")
            response_parts.append(f"   - **Service:** {service}\n")
            response_parts.append(f"   - **Message:**\n   ```\n   {message}\n   ```\n\n")
        
        # Add analysis summary
        response_parts.append("\n---\n### 📊 Quick Analysis\n")
        
        # Check severity distribution
        severities = [str(log.get('severity', 'INFO')).upper() for log in logs if log.get('severity') is not None]
        error_count = sum(1 for s in severities if s in ['ERROR', 'CRITICAL', 'FATAL', '3', '4'])
        warn_count = sum(1 for s in severities if s in ['WARNING', 'WARN', '2'])
        
        if error_count > 0:
            response_parts.append(f"- ⚠️ **{error_count}** of these logs are errors/critical\n")
        if warn_count > 0:
            response_parts.append(f"- ⚡ **{warn_count}** are warnings\n")
        
        # Provide context-aware tips based on what they searched
        user_lower = user_message.lower()
        if 'login' in user_lower or 'authentication' in user_lower or 'password' in user_lower:
            response_parts.append("\n**💡 Security Tip:** Failed login attempts may indicate brute-force attacks. Consider:\n")
            response_parts.append("- Implementing rate limiting\n")
            response_parts.append("- Adding CAPTCHA after multiple failures\n")
            response_parts.append("- Setting up account lockout policies\n")
        elif 'timeout' in user_lower or 'connection' in user_lower:
            response_parts.append("\n**💡 Troubleshooting Tip:** Connection/timeout issues often indicate:\n")
            response_parts.append("- Network congestion or firewall issues\n")
            response_parts.append("- Overloaded servers needing scaling\n")
            response_parts.append("- DNS resolution problems\n")
        elif 'error' in user_lower or 'exception' in user_lower:
            response_parts.append("\n**💡 Debugging Tip:** To investigate these errors:\n")
            response_parts.append("- Check stack traces for root cause\n")
            response_parts.append("- Look at logs from just before the error\n")
            response_parts.append("- Verify external service connectivity\n")
    else:
        response_parts.append("\n❌ **No matching logs found.**\n")
        response_parts.append("\n**Suggestions:**\n")
        response_parts.append("- Try different search terms\n")
        response_parts.append("- Check if logs have been ingested recently\n")
        response_parts.append(f"- The search looked for: {terms_str}\n")
    
    return ''.join(response_parts)


async def _run_ml_analysis(message: str, detector, classifier, security) -> Dict[str, Any]:
    """Run all ML models on a message and return structured analysis"""
    log_entry = {"message": message, "severity": "INFO"}
    
    try:
        anomalies = await detector.detect([log_entry])
        classifications = await classifier.classify([log_entry])
        threats = await security.detect([log_entry])
        
        a = anomalies[0]
        c = classifications[0]
        t = threats[0]
        
        return {
            "input": message,
            "anomaly": {
                "is_anomaly": bool(a.is_anomaly),
                "score": round(float(a.anomaly_score), 3),
                "type": str(a.anomaly_type),
                "explanation": str(a.explanation) if a.is_anomaly else None
            },
            "classification": {
                "category": str(c.category),
                "domain": str(c.domain),
                "severity": str(c.severity),
                "confidence": round(float(c.confidence), 2)
            },
            "security": {
                "is_threat": bool(t.is_threat),
                "score": round(float(t.threat_score), 3),
                "type": str(t.threat_type) if t.threat_type != "none" else None,
                "severity": str(t.severity) if t.is_threat else None,
                "actions": list(t.recommended_actions[:3]) if t.is_threat and t.recommended_actions else []
            }
        }
    except Exception as e:
        logger.error(f"ML analysis failed: {e}")
        return {"error": str(e)}


async def _generate_llm_response(user_message: str, ml_analysis: Dict[str, Any], 
                                  context: Optional[List[str]] = None) -> Optional[str]:
    """Generate intelligent response using LLM with timeout - includes log search"""
    
    # ===== CHECK IF USER WANTS TO SEARCH LOGS =====
    # If it's a log search query, search the database first
    log_context = ""
    log_results = []
    if _is_log_search_query(user_message):
        service_filter = _detect_service_filter(user_message)
        search_terms = _extract_search_terms(user_message)
        if search_terms or service_filter:
            log_results = await _search_logs_in_database(search_terms, limit=15, service_filter=service_filter)
            if log_results:
                # Format logs for LLM context
                log_context = f"\n\n=== ACTUAL LOGS FROM DATABASE ({len(log_results)} found) ===\n"
                for i, log in enumerate(log_results[:15], 1):  # Send up to 15 to LLM
                    severity = str(log.get('severity', 'INFO')).upper() if log.get('severity') else 'INFO'
                    message = log.get('message', '')[:300]  # Truncate for prompt size
                    log_context += f"{i}. [{severity}] {message}\n"
                log_context += "=== END OF LOGS ===\n"
    
    ollama = get_ollama_client()
    
    # Build a concise prompt
    a = ml_analysis['anomaly']
    c = ml_analysis['classification']
    s = ml_analysis['security']
    
    prompt = f"""User Question: {user_message[:200]}
{log_context}
ML Analysis: Anomaly={a['is_anomaly']} (score:{a['score']}), Category={c['category']}, Severity={c['severity']}, Threat={s['is_threat']} ({s['type'] or 'none'})

{f"The user is asking about logs. I found {len(log_results) if log_context else 0} matching logs above. Include specific examples from these logs in your response." if log_context else ""}

Provide a helpful, detailed response. If logs were found, show specific examples with timestamps and details."""

    messages = [{"role": "user", "content": prompt}]
    
    try:
        # Add timeout to prevent hanging
        response = await asyncio.wait_for(
            ollama.chat(
                messages=messages,
                system=LOG_EXPERT_SYSTEM_PROMPT,
                temperature=0.5  # Lower temperature = faster
            ),
            timeout=LLM_TIMEOUT
        )
        
        # If we have log results, append them to the LLM response for clarity
        if log_context and log_results:
            response += f"\n\n---\n\n## 📋 Full Log Results ({len(log_results)} entries)\n\n"
            for i, log in enumerate(log_results, 1):
                # Convert numeric severity to string labels
                raw_severity = log.get('severity', 'INFO')
                severity_map = {
                    '0': 'DEBUG', '1': 'INFO', '2': 'WARNING', '3': 'ERROR', '4': 'CRITICAL',
                    0: 'DEBUG', 1: 'INFO', 2: 'WARNING', 3: 'ERROR', 4: 'CRITICAL',
                }
                severity = severity_map.get(raw_severity, str(raw_severity).upper() if raw_severity else 'INFO')
                severity_icon = {'ERROR': '🔴', 'WARNING': '🟡', 'WARN': '🟡', 'INFO': '🔵', 'DEBUG': '⚪', 
                                'CRITICAL': '🟣'}.get(severity, '⚪')
                message = log.get('message', '')[:250]
                timestamp = log.get('timestamp', 'N/A')
                service = log.get('service', 'unknown')
                response += f"**{i}.** {severity_icon} `{severity}` | {timestamp} | {service}\n```\n{message}\n```\n\n"
        
        return response
    except asyncio.TimeoutError:
        logger.warning(f"LLM timeout after {LLM_TIMEOUT}s")
        return None
    except OllamaError as e:
        logger.warning(f"LLM not available: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return None


async def _generate_fallback_response(user_message: str, ml_analysis: Dict[str, Any]) -> str:
    """Generate intelligent answers to ANY question using ML analysis as context"""
    
    a = ml_analysis['anomaly']
    c = ml_analysis['classification']
    s = ml_analysis['security']
    user_lower = user_message.lower()
    
    # ===== CHECK IF USER WANTS TO SEARCH LOGS =====
    if _is_log_search_query(user_message):
        service_filter = _detect_service_filter(user_message)
        search_terms = _extract_search_terms(user_message)
        if search_terms or service_filter:
            log_results = await _search_logs_in_database(search_terms, service_filter=service_filter)
            if log_results:
                return _format_log_search_results(user_message, search_terms, log_results, a, c, s)
    
    # ===== INTELLIGENT QUESTION UNDERSTANDING =====
    # Detect what the user is asking
    is_question = '?' in user_message or any(w in user_lower for w in ['what', 'why', 'how', 'when', 'where', 'who', 'is this', 'can you', 'could you', 'tell me', 'explain'])
    
    # Question type detection
    asks_why = any(w in user_lower for w in ['why', 'cause', 'reason', 'explain why', 'what caused', 'root cause'])
    asks_what = any(w in user_lower for w in ['what is', 'what does', 'what happened', 'what\'s'])
    asks_how = any(w in user_lower for w in ['how to', 'how do', 'how can', 'how should'])
    asks_fix = any(w in user_lower for w in ['fix', 'solve', 'resolve', 'prevent', 'stop', 'protect', 'mitigate'])
    asks_safe = any(w in user_lower for w in ['safe', 'secure', 'dangerous', 'risky', 'threat', 'attack', 'malicious', 'vulnerability'])
    asks_meaning = any(w in user_lower for w in ['mean', 'means', 'meaning', 'indicate', 'indicates', 'signify'])
    asks_severity = any(w in user_lower for w in ['severe', 'severity', 'serious', 'critical', 'urgent', 'important', 'priority'])
    asks_action = any(w in user_lower for w in ['do', 'action', 'step', 'next', 'should i', 'recommend'])
    
    # Log type detection from user's message
    mentions_error = any(w in user_lower for w in ['error', 'exception', 'fail', 'crash', 'down', 'broken'])
    mentions_timeout = any(w in user_lower for w in ['timeout', 'slow', 'latency', 'delay', 'hang', 'stuck'])
    mentions_auth = any(w in user_lower for w in ['login', 'auth', 'password', 'credential', 'access denied', 'permission'])
    mentions_network = any(w in user_lower for w in ['network', 'connection', 'socket', 'port', 'dns', 'ip', 'http'])
    mentions_database = any(w in user_lower for w in ['database', 'db', 'sql', 'query', 'mysql', 'postgres', 'mongo'])
    mentions_memory = any(w in user_lower for w in ['memory', 'ram', 'heap', 'oom', 'out of memory', 'leak'])
    mentions_disk = any(w in user_lower for w in ['disk', 'storage', 'space', 'full', 'inode', 'filesystem'])
    
    # ===== BUILD INTELLIGENT RESPONSE =====
    
    # PRIORITY 1: How-to and prevention questions (even if they mention security terms)
    if asks_how or asks_fix:
        return _answer_solution_question(user_message, user_lower, a, c, s)
    
    # PRIORITY 2: If an actual threat was detected in the input
    if s['is_threat']:
        return _answer_security_question(user_message, user_lower, a, c, s, asks_why, asks_how, asks_fix)
    
    # PRIORITY 3: If asking about an error/crash/issue
    if asks_why or asks_what or mentions_error or mentions_timeout:
        return _answer_diagnostic_question(user_message, user_lower, a, c, s, 
                                           mentions_timeout, mentions_auth, mentions_network, 
                                           mentions_database, mentions_memory, mentions_disk)
    
    # PRIORITY 4: Security-related questions (is this safe, etc)
    if asks_safe:
        return _answer_security_question(user_message, user_lower, a, c, s, asks_why, asks_how, asks_fix)
    
    # If asking what something means
    if asks_meaning:
        return _answer_meaning_question(user_message, user_lower, a, c, s)
    
    # If asking about severity/priority
    if asks_severity:
        return _answer_severity_question(user_message, a, c, s)
    
    # If it's a question but we're not sure what type
    if is_question:
        return _answer_general_question(user_message, user_lower, a, c, s)
    
    # Default: treat as a log to analyze
    return _analyze_log_entry(user_message, a, c, s)


def _answer_security_question(user_message: str, user_lower: str, a: dict, c: dict, s: dict,
                               asks_why: bool, asks_how: bool, asks_fix: bool) -> str:
    """Answer questions about security threats"""
    
    if s['is_threat']:
        threat_type = s['type'] or 'suspicious_pattern'
        score = s['score']
        severity = s['severity'] or 'medium'
        
        response = f"""# 🔒 Security Analysis

**Your input has been analyzed for security threats.**

---

## 🚨 Threat Detected: {threat_type.replace('_', ' ').title()}

| Metric | Value |
|--------|-------|
| **Threat Score** | `{score:.2f}` (0-1 scale) |
| **Severity** | `{severity.upper()}` |
| **Classification** | `{c['category']}` |

---

"""
        if asks_why:
            response += """## ❓ Why is this flagged as a threat?

This input matches patterns commonly associated with security attacks:
- Contains syntax or characters used in exploitation attempts
- Matches signatures from known attack patterns
- Shows characteristics of malicious payloads

The ML model was trained on thousands of attack examples and detected similar patterns in your input.

---

"""
        
        if asks_how or asks_fix:
            response += """## 🛡️ How to Protect Against This

1. **Input Validation** - Always validate and sanitize user input
2. **Parameterized Queries** - Never concatenate user input into queries
3. **Encoding** - Properly encode output based on context
4. **WAF** - Deploy a Web Application Firewall
5. **Security Headers** - Implement proper security headers
6. **Regular Updates** - Keep all software up to date

---

"""
        
        if s['actions']:
            response += "## ⚡ Recommended Actions\n\n"
            for i, action in enumerate(s['actions'], 1):
                response += f"{i}. {action}\n"
        
        return response
    
    else:
        return f"""# ✅ Security Check Complete

**Good news! No security threats detected in your input.**

---

## Analysis Results

| Check | Status |
|-------|--------|
| **Threat Detection** | ✅ Clean |
| **Threat Score** | `{s['score']:.2f}` (low = safe) |
| **Classification** | `{c['category']}` |
| **Severity** | `{c['severity']}` |

---

## What was checked:
- SQL Injection patterns
- Cross-Site Scripting (XSS)
- Command Injection
- Path Traversal
- NoSQL Injection
- SSTI, XXE, SSRF patterns
- And many more...

Your input appears to be safe and doesn't match known attack signatures.
"""


def _answer_diagnostic_question(user_message: str, user_lower: str, a: dict, c: dict, s: dict,
                                  mentions_timeout: bool, mentions_auth: bool, mentions_network: bool,
                                  mentions_database: bool, mentions_memory: bool, mentions_disk: bool) -> str:
    """Answer questions about errors, crashes, and diagnostics"""
    
    response = f"""# 🔍 Diagnostic Analysis

**Let me analyze this issue for you.**

---

## 📊 What the ML Model Detected

| Analysis | Result |
|----------|--------|
| **Category** | `{c['category']}` |
| **Severity** | `{c['severity']}` |
| **Domain** | `{c['domain']}` |
| **Anomaly Score** | `{a['score']:.2f}` |
| **Is Anomaly** | {'⚠️ Yes' if a['is_anomaly'] else '✅ No'} |

---

## 🤔 Possible Causes

"""
    
    # Context-aware troubleshooting based on what they mentioned
    if mentions_timeout or 'timeout' in user_lower:
        response += """### Timeout/Latency Issues:

1. **Server Overload** - CPU or memory exhaustion
2. **Network Latency** - Slow network path or packet loss
3. **Database Bottleneck** - Slow queries or connection pool exhausted
4. **External Service** - Third-party API not responding
5. **Resource Starvation** - Thread pool or connection limits reached

**Quick Checks:**
- `top` / `htop` - Check CPU/memory usage
- `netstat -an | grep ESTABLISHED` - Check active connections
- Check application logs for slow query warnings

---

"""
    
    if mentions_auth or 'login' in user_lower or 'password' in user_lower:
        response += """### Authentication Issues:

1. **Invalid Credentials** - Wrong username/password
2. **Account Locked** - Too many failed attempts
3. **Token Expired** - Session or JWT expired
4. **Permission Denied** - User lacks required role
5. **Configuration Error** - Auth service misconfigured

**Quick Checks:**
- Verify the account exists and is active
- Check for account lockout policies
- Review auth service logs
- Verify token expiration settings

---

"""
    
    if mentions_network or 'connection' in user_lower:
        response += """### Network/Connection Issues:

1. **DNS Resolution Failure** - Cannot resolve hostname
2. **Firewall Blocking** - Port blocked by firewall
3. **Service Down** - Target service not running
4. **Port Exhaustion** - Too many connections
5. **Network Partition** - Network path unavailable

**Quick Checks:**
- `ping <host>` - Check basic connectivity
- `telnet <host> <port>` - Check port accessibility
- `nslookup <host>` - Check DNS resolution
- Check firewall rules and security groups

---

"""
    
    if mentions_database or 'sql' in user_lower or 'query' in user_lower:
        response += """### Database Issues:

1. **Connection Pool Exhausted** - Too many concurrent connections
2. **Slow Query** - Query taking too long to execute
3. **Deadlock** - Transactions blocking each other
4. **Disk Full** - Database storage exhausted
5. **Index Missing** - Query not using proper indexes

**Quick Checks:**
- Check database connection pool metrics
- Review slow query log
- Check disk space on database server
- Analyze query execution plans

---

"""
    
    if mentions_memory or 'memory' in user_lower or 'oom' in user_lower:
        response += """### Memory Issues:

1. **Memory Leak** - Application not releasing memory
2. **Heap Exhaustion** - JVM/process heap too small
3. **OOM Killer** - Linux killed process due to memory pressure
4. **Large Payload** - Processing very large data
5. **Cache Growth** - In-memory cache unbounded

**Quick Checks:**
- `free -h` - Check system memory
- Check application heap settings
- Review memory profiler output
- Look for OOM messages in dmesg

---

"""
    
    if mentions_disk or 'disk' in user_lower or 'storage' in user_lower:
        response += """### Disk/Storage Issues:

1. **Disk Full** - No free space remaining
2. **Inode Exhaustion** - Too many small files
3. **I/O Bottleneck** - Slow disk throughput
4. **Permission Issues** - Cannot write to directory
5. **Log Growth** - Logs filling up disk

**Quick Checks:**
- `df -h` - Check disk usage
- `df -i` - Check inode usage
- `du -sh /var/log/*` - Check log sizes
- Consider log rotation

---

"""
    
    # If they didn't mention anything specific, give general guidance
    if not any([mentions_timeout, mentions_auth, mentions_network, mentions_database, mentions_memory, mentions_disk]):
        response += """### General Troubleshooting Steps:

1. **Check the logs** - Look for error messages around the time of the issue
2. **Verify resources** - CPU, memory, disk, network
3. **Recent changes** - Any deployments or config changes?
4. **External dependencies** - Are all services up?
5. **Pattern analysis** - Is this recurring at specific times?

---

"""
    
    response += """## 💡 Need More Help?

Provide more context like:
- Full error messages
- Timestamps
- What changed recently
- System metrics

The more context, the better I can help diagnose the issue!
"""
    
    return response


def _answer_solution_question(user_message: str, user_lower: str, a: dict, c: dict, s: dict) -> str:
    """Answer how-to and solution questions - especially security prevention"""
    
    # Security prevention knowledge base
    security_solutions = {
        'sql injection': {
            'title': 'SQL Injection Prevention',
            'description': 'SQL injection attacks insert malicious SQL code into queries to manipulate databases.',
            'prevention': [
                '✅ **Use Parameterized Queries** - Never concatenate user input into SQL',
                '✅ **Prepared Statements** - Use your framework\'s built-in protection',
                '✅ **Input Validation** - Whitelist allowed characters',
                '✅ **Least Privilege** - Database accounts with minimal permissions',
                '✅ **WAF** - Deploy Web Application Firewall rules',
                '✅ **ORM** - Use Object-Relational Mapping instead of raw SQL'
            ],
            'code': '''# Python with parameterized query
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))

# Node.js with prepared statements
db.query("SELECT * FROM users WHERE id = ?", [userId])'''
        },
        'xss': {
            'title': 'Cross-Site Scripting (XSS) Prevention',
            'description': 'XSS attacks inject malicious scripts into web pages viewed by other users.',
            'prevention': [
                '✅ **Output Encoding** - HTML-encode all user-generated content',
                '✅ **Content Security Policy** - Restrict script sources',
                '✅ **HttpOnly Cookies** - Prevent JavaScript cookie access',
                '✅ **Input Sanitization** - Strip dangerous characters',
                '✅ **Use Frameworks** - Modern frameworks auto-escape by default'
            ],
            'code': '''// Use textContent instead of innerHTML
element.textContent = userInput;

// React auto-escapes
<div>{userInput}</div>

// CSP header
Content-Security-Policy: script-src 'self\''''
        },
        'command injection': {
            'title': 'Command Injection Prevention',
            'description': 'Command injection executes arbitrary system commands on the host.',
            'prevention': [
                '✅ **Avoid System Calls** - Use language-native libraries',
                '✅ **Input Validation** - Strict whitelist of allowed characters',
                '✅ **No Shell=True** - Avoid shell interpretation',
                '✅ **Sandboxing** - Run processes with minimal privileges',
                '✅ **Use Arrays** - Pass commands as arrays, not strings'
            ],
            'code': '''# Python - SAFE
subprocess.run(['ls', '-la', directory], shell=False)

# Python - DANGEROUS
subprocess.run(f'ls -la {directory}', shell=True)'''
        },
        'path traversal': {
            'title': 'Path Traversal Prevention',
            'description': 'Path traversal accesses files outside intended directories.',
            'prevention': [
                '✅ **Canonicalize Paths** - Resolve to absolute paths',
                '✅ **Validate Against Basepath** - Ensure path stays within allowed directory',
                '✅ **Use IDs** - Reference files by ID, not path',
                '✅ **Chroot/Jail** - Restrict file system access',
                '✅ **Strip Traversal** - Remove ../ sequences'
            ],
            'code': '''# Python - validate path
import os
base = '/var/www/uploads'
requested = os.path.realpath(os.path.join(base, user_path))
if not requested.startswith(base):
    raise SecurityError("Path traversal detected")'''
        },
        'nosql injection': {
            'title': 'NoSQL Injection Prevention',
            'description': 'NoSQL injection exploits MongoDB and similar databases using operators like $gt, $ne.',
            'prevention': [
                '✅ **Input Validation** - Reject $ prefixed keys in input',
                '✅ **Type Checking** - Ensure expected data types',
                '✅ **Use ODM/ORM** - Mongoose, etc. with proper validation',
                '✅ **Disable $where** - Turn off JavaScript execution',
                '✅ **Schema Validation** - Enforce document structure'
            ],
            'code': '''// Mongoose with schema validation
const userSchema = new Schema({
  username: { type: String, required: true },
  password: { type: String, required: true }
});

// Sanitize input - reject $ operators
if (JSON.stringify(input).includes(\'$\')) {
  throw new Error(\'Invalid input\');
}'''
        },
        'log4shell': {
            'title': 'Log4Shell (Log4j RCE) Prevention',
            'description': 'Log4Shell (CVE-2021-44228) allows RCE via JNDI lookups in log messages.',
            'prevention': [
                '✅ **Update Log4j** - Upgrade to 2.17.1 or higher',
                '✅ **Disable Lookups** - Set log4j2.formatMsgNoLookups=true',
                '✅ **Remove JndiLookup** - Delete class from classpath',
                '✅ **WAF Rules** - Block ${jndi: patterns',
                '✅ **Egress Filtering** - Block outbound LDAP/RMI'
            ],
            'code': '''# Java - disable lookups
-Dlog4j2.formatMsgNoLookups=true

# Remove JndiLookup class
zip -q -d log4j-core-*.jar org/apache/logging/log4j/core/lookup/JndiLookup.class'''
        },
        'ssrf': {
            'title': 'Server-Side Request Forgery (SSRF) Prevention',
            'description': 'SSRF tricks servers into making requests to internal resources.',
            'prevention': [
                '✅ **Whitelist URLs** - Only allow known external hosts',
                '✅ **Block Internal IPs** - Deny 127.0.0.1, 10.x, 169.254.x',
                '✅ **Disable Redirects** - Don\'t follow HTTP redirects',
                '✅ **IMDSv2** - Require tokens for cloud metadata',
                '✅ **Network Segmentation** - Isolate server networks'
            ],
            'code': '''# Python - validate URL
from urllib.parse import urlparse
import ipaddress

def is_safe_url(url):
    parsed = urlparse(url)
    try:
        ip = ipaddress.ip_address(parsed.hostname)
        return ip.is_global  # Only allow public IPs
    except ValueError:
        return parsed.hostname in ALLOWED_HOSTS'''
        },
        'ssti': {
            'title': 'Server-Side Template Injection Prevention',
            'description': 'SSTI exploits template engines (Jinja2, Twig) for RCE.',
            'prevention': [
                '✅ **Sandbox Templates** - Use restricted environments',
                '✅ **Never Embed Input** - Don\'t put user input in templates',
                '✅ **Use Static Templates** - Pass data as context only',
                '✅ **Autoescape** - Enable automatic escaping',
                '✅ **Template Hardening** - Disable dangerous functions'
            ],
            'code': '''# Jinja2 - SAFE (data as context)
render_template("page.html", name=user_input)

# Jinja2 - DANGEROUS (user input in template)
Template(user_input).render()  # NEVER DO THIS'''
        },
        'xxe': {
            'title': 'XML External Entity (XXE) Prevention',
            'description': 'XXE exploits XML parsers to read files or perform SSRF.',
            'prevention': [
                '✅ **Disable DTDs** - Set disallow-doctype-decl',
                '✅ **Disable External Entities** - Configure parser safely',
                '✅ **Use JSON** - Prefer JSON over XML',
                '✅ **Update Libraries** - Use patched XML parsers',
                '✅ **Input Validation** - Filter DOCTYPE declarations'
            ],
            'code': '''# Python lxml - disable entities
from lxml import etree
parser = etree.XMLParser(resolve_entities=False, no_network=True)

# Java - secure parsing
factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);'''
        }
    }
    
    # Detect which attack type the user is asking about
    detected_attack = None
    for attack, info in security_solutions.items():
        if attack in user_lower:
            detected_attack = attack
            break
    
    if detected_attack:
        info = security_solutions[detected_attack]
        response = f"""# 🛡️ {info['title']}

{info['description']}

---

## ✅ Prevention Measures

"""
        for item in info['prevention']:
            response += f"{item}\n"
        
        response += f"""
---

## 💻 Secure Code Example

```
{info['code']}
```

---

## 📚 Additional Resources

- OWASP Cheat Sheet for {detected_attack.replace(' ', '-').title()}
- Security testing with automated scanners
- Regular code reviews for security issues

---

💡 **More Questions?** Ask about other attack types like:
- XSS, Command Injection, Path Traversal
- NoSQL Injection, Log4Shell, SSRF, SSTI, XXE
"""
        return response
    
    # Generic solution response if no specific attack detected
    response = """# 🛠️ Solutions & Recommendations

Based on your question, here are some approaches:

---

"""
    
    if s['is_threat']:
        response += f"""## 🔒 Security Remediation

Since a **{s['type'] or 'security threat'}** was detected:

### Immediate Actions:
1. **Block the Source** - If from a specific IP, add to blocklist
2. **Review Logs** - Check for similar patterns
3. **Assess Impact** - Determine if any data was compromised
4. **Patch Vulnerabilities** - Update affected components

### Long-term Prevention:
1. Input validation and sanitization
2. Implement Web Application Firewall (WAF)
3. Regular security audits
4. Security awareness training
5. Incident response plan

---

"""
    
    if a['is_anomaly']:
        response += f"""## ⚠️ Addressing the Anomaly

This log was flagged as unusual (score: {a['score']:.2f}).

### Investigation Steps:
1. **Find Related Logs** - Check logs before/after this event
2. **Identify Pattern** - Is this recurring or one-time?
3. **Check Changes** - Any recent deployments or changes?
4. **Verify Health** - Are all services functioning normally?

### If it's a real issue:
1. Identify the root cause
2. Implement a fix
3. Add monitoring/alerting
4. Document for future reference

### If it's expected behavior:
1. Update the ML model with this pattern
2. Add to known-good patterns
3. Adjust detection thresholds

---

"""
    
    response += """## 📚 General Best Practices

1. **Logging** - Ensure comprehensive, structured logging
2. **Monitoring** - Set up alerts for critical metrics
3. **Runbooks** - Document common issues and solutions
4. **Automation** - Automate repetitive troubleshooting
5. **Regular Review** - Periodically review and tune alerts

---

## 💬 Ask Me More!

I can help with security questions like:
- "How do I prevent SQL injection?"
- "What's the best way to stop XSS attacks?"
- "How to fix command injection vulnerabilities?"
- "How to prevent SSRF attacks?"
"""
    
    return response


def _answer_meaning_question(user_message: str, user_lower: str, a: dict, c: dict, s: dict) -> str:
    """Explain what something means"""
    
    return f"""# 📖 Understanding Your Log

**Let me explain what this means.**

---

## 📊 Classification

Your input was classified as:

| Attribute | Value | Meaning |
|-----------|-------|---------|
| **Category** | `{c['category']}` | The type of log entry |
| **Severity** | `{c['severity']}` | How critical this is |
| **Domain** | `{c['domain']}` | The system area it relates to |
| **Confidence** | `{c['confidence']:.0%}` | How sure the model is |

---

## 🔍 What This Tells Us

### Severity Levels Explained:
- **DEBUG** - Detailed diagnostic info for developers
- **INFO** - Normal operational messages
- **WARNING** - Something unusual but not an error
- **ERROR** - Something went wrong, needs attention
- **CRITICAL** - Severe problem, immediate action needed

### Your Severity: `{c['severity']}`
{
"This is informational - normal system operation." if c['severity'] == 'INFO' else
"This is a warning - worth investigating but not urgent." if c['severity'] == 'WARNING' else
"This is an error - something failed and needs attention." if c['severity'] == 'ERROR' else
"This is critical - requires immediate action!" if c['severity'] == 'CRITICAL' else
"Debug information for troubleshooting."
}

---

## 📈 Anomaly Status

- **Score**: `{a['score']:.2f}` (0 = normal, 1 = highly unusual)
- **Status**: {'⚠️ Anomalous - deviates from normal patterns' if a['is_anomaly'] else '✅ Normal - matches expected patterns'}

{f"**Why anomalous:** {a['explanation']}" if a['is_anomaly'] and a['explanation'] else ""}

---

## 🔒 Security Status

- **Threat Detected**: {'🚨 Yes' if s['is_threat'] else '✅ No'}
- **Threat Score**: `{s['score']:.2f}`
{f"- **Threat Type**: {s['type']}" if s['is_threat'] else ""}

---

💡 **Need more explanation?** Ask me about specific terms or concepts!
"""


def _answer_severity_question(user_message: str, a: dict, c: dict, s: dict) -> str:
    """Answer questions about severity and priority"""
    
    # Determine overall severity
    if s['is_threat']:
        overall = "HIGH - Security threat detected"
        priority = "🔴 Immediate"
        action = "Investigate immediately, potential security incident"
    elif c['severity'] in ['CRITICAL', 'ERROR']:
        overall = "HIGH - Error condition"
        priority = "🔴 Urgent"
        action = "Address as soon as possible"
    elif a['is_anomaly'] and a['score'] > 0.7:
        overall = "MEDIUM-HIGH - Significant anomaly"
        priority = "🟠 High"
        action = "Investigate within hours"
    elif c['severity'] == 'WARNING' or a['is_anomaly']:
        overall = "MEDIUM - Warning/Minor anomaly"
        priority = "🟡 Normal"
        action = "Review when convenient"
    else:
        overall = "LOW - Normal operation"
        priority = "🟢 Low"
        action = "No action required"
    
    return f"""# 📊 Severity Assessment

---

## Overall Assessment

| Metric | Rating |
|--------|--------|
| **Overall Severity** | **{overall}** |
| **Priority** | {priority} |
| **Recommended Action** | {action} |

---

## Detailed Breakdown

### Classification Severity: `{c['severity']}`
"""+ ("""
- **CRITICAL/ERROR**: Immediate attention required
""" if c['severity'] in ['CRITICAL', 'ERROR'] else """
- **WARNING**: Should be reviewed
""" if c['severity'] == 'WARNING' else """
- **INFO/DEBUG**: Normal operation
""") + f"""

### Anomaly Severity
- **Score**: `{a['score']:.2f}`
- **Threshold**: 0.5 (scores above are flagged)
- **Status**: {'⚠️ Above threshold - unusual' if a['is_anomaly'] else '✅ Normal'}

### Security Severity
- **Threat Detected**: {'🚨 Yes' if s['is_threat'] else '✅ No'}
- **Threat Score**: `{s['score']:.2f}`
{f"- **Threat Severity**: `{s['severity']}`" if s['is_threat'] else ""}

---

## 📋 Priority Matrix

| If you see... | Priority |
|---------------|----------|
| Security threat | 🔴 Immediate |
| Critical/Error + Anomaly | 🔴 Urgent |
| Error or High anomaly | 🟠 High |
| Warning | 🟡 Normal |
| Info/Debug | 🟢 Low |

---

**Your case falls into: {priority}**
"""


def _answer_general_question(user_message: str, user_lower: str, a: dict, c: dict, s: dict) -> str:
    """Answer general questions using available context"""
    
    return f"""# 💬 Let Me Help You

I analyzed your message and here's what I found:

---

## 📊 Analysis Results

| Metric | Value |
|--------|-------|
| **Category** | `{c['category']}` |
| **Severity** | `{c['severity']}` |
| **Anomaly Score** | `{a['score']:.2f}` {'⚠️' if a['is_anomaly'] else '✅'} |
| **Threat Score** | `{s['score']:.2f}` {'🚨' if s['is_threat'] else '✅'} |

---

## 🤔 How Can I Help?

I can answer questions like:

**About Security:**
- "Is this safe?"
- "Is this a SQL injection?"
- "How do I prevent XSS attacks?"

**About Errors:**
- "Why is my server crashing?"
- "What causes timeout errors?"
- "How do I fix connection issues?"

**About Logs:**
- "What does this error mean?"
- "Is this severity critical?"
- "Should I be worried about this?"

**About Solutions:**
- "How do I fix this?"
- "What should I do next?"
- "How can I prevent this?"

---

## 💡 Tips for Better Answers

1. **Be specific** - Include the actual error message
2. **Add context** - What were you doing when it happened?
3. **Ask directly** - "Why did X happen?" or "How do I fix Y?"

---

**Try asking me something specific!**
"""


def _analyze_log_entry(user_message: str, a: dict, c: dict, s: dict) -> str:
    """Analyze a log entry that isn't a question - provide comprehensive analysis"""
    
    # Security threat detected
    if s['is_threat']:
        threat_type = (s['type'] or 'suspicious_pattern').replace('_', ' ').title()
        return f"""# 🚨 Security Threat Detected

## ⚠️ {threat_type}

| Metric | Value |
|--------|-------|
| **Threat Score** | `{s['score']:.2f}` |
| **Severity** | `{(s['severity'] or 'medium').upper()}` |
| **Classification** | `{c['category']}` |

---

## 🔍 Analysis

This input contains patterns commonly associated with **{threat_type.lower()}** attacks.
The ML security model detected suspicious signatures that match known attack patterns.

---

## 🛡️ What to Do

1. **Don't execute** this input in any system
2. **Block the source** if this came from an external request
3. **Log and alert** - Document this attempt for security review
4. **Review similar inputs** - Check for related attack patterns

---

{f"## ⚡ Recommended Actions" + chr(10) + chr(10) + chr(10).join(f"- {a}" for a in s['actions']) if s['actions'] else ""}

---

💡 **Ask me more:** "How do I prevent {threat_type.lower()}?" or "What does this attack do?"
"""
    
    # Anomaly detected
    if a['is_anomaly']:
        return f"""# ⚠️ Anomaly Detected

**This log entry deviates from normal patterns.**

---

## 📊 Analysis Results

| Metric | Value |
|--------|-------|
| **Anomaly Score** | `{a['score']:.3f}` (1.0 = most anomalous) |
| **Category** | `{c['category']}` |
| **Severity** | `{c['severity']}` |
| **Domain** | `{c['domain']}` |

---

## 🔍 What This Means

{a['explanation'] if a['explanation'] else 'This pattern differs significantly from typical logs in your training data.'}

**Possible reasons:**
- 🆕 New type of event not seen during training
- 🐛 Application error or unexpected behavior
- 🔧 Configuration change affecting log format
- ⏰ Time-based variation (off-hours activity)

---

## 📋 Recommended Actions

1. **Investigate context** - Check logs before/after this entry
2. **Verify impact** - Did this affect users or services?
3. **Identify root cause** - What triggered this unusual pattern?
4. **Update baseline** - If expected, retrain models with this pattern

---

💡 **Ask me:** "Why is this unusual?" or "How do I investigate this?"
"""
    
    # Normal log - provide classification details
    return f"""# ✅ Log Analysis Complete

**This appears to be a normal log entry.**

---

## 📊 Classification Results

| Attribute | Value |
|-----------|-------|
| **Category** | `{c['category']}` |
| **Severity** | `{c['severity']}` |
| **Domain** | `{c['domain']}` |
| **Confidence** | `{c['confidence']:.0%}` |
| **Anomaly Score** | `{a['score']:.3f}` (low = normal) |

---

## 🛡️ Security Status

✅ **No threats detected** - Input doesn't match known attack patterns.

---

## 📝 Log Preview

```
{user_message[:300]}{'...' if len(user_message) > 300 else ''}
```

---

## 💡 You Can Ask Me

- "What does this log mean?"
- "Should I be concerned about this?"
- "What caused this error?"
- "How do I fix [issue]?"
- "Is this a security threat?"

I can answer **any question** about logs, errors, security, and troubleshooting!
"""


# ===== ML CHAT ENDPOINT =====
@router.post("/chat", response_model=MLChatResponse)
async def ml_chat(request: MLChatRequest):
    """
    Intelligent AI Chat for log analysis.
    
    This endpoint combines ML model analysis with LLM understanding to:
    - Answer ANY question about logs
    - Explain anomalies and their causes
    - Detect and explain security threats
    - Provide troubleshooting advice
    - Have natural conversations about log data
    
    Examples:
    - "What caused this server crash?"
    - "Is this log message a security threat?"
    - "Why is my database timing out?"
    - "Explain this error: [error message]"
    - "What does this log mean?"
    """
    user_message = request.message.strip()
    
    # Get ML models
    detector = await get_anomaly_detector()
    classifier = await get_log_classifier()
    security = await get_security_detector()
    
    model_status = {
        "anomaly_detector": bool(detector.is_trained),
        "log_classifier": bool(classifier.is_trained),
        "security_detector": bool(security.is_trained),
    }
    
    all_trained = all(model_status.values())
    
    # Check if models are trained
    if not all_trained:
        not_trained = [k.replace('_', ' ').title() for k, v in model_status.items() if not v]
        return MLChatResponse(
            response=f"⚠️ **Models Not Trained**\n\nI need trained ML models to analyze logs effectively.\n\n"
                     f"**Missing:** {', '.join(not_trained)}\n\n"
                     "Please train the models first using the AI Dashboard or `/ml/train` endpoint.",
            analysis=None,
            model_status=model_status,
            suggestions=["Train models", "Check /ml/train/status"]
        )
    
    # Run ML analysis on the user's input
    ml_analysis = await _run_ml_analysis(user_message, detector, classifier, security)
    
    if "error" in ml_analysis:
        return MLChatResponse(
            response=f"❌ **Analysis Error**\n\n{ml_analysis['error']}",
            analysis=None,
            model_status=model_status,
            suggestions=["Try again", "Check model status"]
        )
    
    # Decide whether to use LLM
    use_llm = request.use_llm and not request.fast_mode
    
    if use_llm:
        # Try to generate intelligent response with LLM (with timeout)
        llm_response = await _generate_llm_response(user_message, ml_analysis, request.context)
        if llm_response:
            response = llm_response
        else:
            # LLM timed out or unavailable - use smart fallback
            response = await _generate_fallback_response(user_message, ml_analysis)
    else:
        # Fast mode - skip LLM entirely
        response = await _generate_fallback_response(user_message, ml_analysis)
    
    # Generate contextual suggestions
    s = ml_analysis['security']
    a = ml_analysis['anomaly']
    
    if s['is_threat']:
        suggestions = ["What should I do about this threat?", "Show me similar attacks", "How to prevent this?"]
    elif a['is_anomaly']:
        suggestions = ["Why is this unusual?", "What could cause this?", "How to investigate?"]
    else:
        suggestions = ["Analyze another log", "Check for security threats", "What patterns do you see?"]
    
    return MLChatResponse(
        response=response,
        analysis=ml_analysis if request.include_raw_analysis else None,
        model_status=model_status,
        suggestions=suggestions
    )


@router.post("/analyze", response_model=Dict[str, Any])
async def full_analysis(request: BatchLogsRequest):
    """
    Perform comprehensive analysis on logs using all ML models.
    
    Returns combined results from:
    - Anomaly detection
    - Classification
    - Security threat detection
    """
    logs = [log.model_dump() for log in request.logs]
    
    # Run all detectors in parallel
    detector = await get_anomaly_detector()
    classifier = await get_log_classifier()
    security = await get_security_detector()
    
    anomalies = await detector.detect(logs)
    classifications = await classifier.classify(logs)
    threats = await security.detect(logs)
    
    # Combine results
    combined_results = []
    for i, log in enumerate(logs):
        combined_results.append({
            "log": log,
            "anomaly": {
                "is_anomaly": anomalies[i].is_anomaly,
                "score": anomalies[i].anomaly_score,
                "type": anomalies[i].anomaly_type,
                "explanation": anomalies[i].explanation
            },
            "classification": {
                "category": classifications[i].category,
                "domain": classifications[i].domain,
                "severity": classifications[i].severity,
                "confidence": classifications[i].confidence
            },
            "security": {
                "is_threat": threats[i].is_threat,
                "score": threats[i].threat_score,
                "type": threats[i].threat_type,
                "actions": threats[i].recommended_actions
            }
        })
    
    # Summary statistics
    summary = {
        "total_logs": len(logs),
        "anomalies_detected": sum(1 for a in anomalies if a.is_anomaly),
        "threats_detected": sum(1 for t in threats if t.is_threat),
        "category_distribution": {},
        "severity_distribution": {}
    }
    
    for c in classifications:
        summary["category_distribution"][c.category] = \
            summary["category_distribution"].get(c.category, 0) + 1
        summary["severity_distribution"][c.severity] = \
            summary["severity_distribution"].get(c.severity, 0) + 1
    
    return {
        "results": combined_results,
        "summary": summary
    }
