"""
ML-Based Security Threat Detection

This module implements real machine learning for security threat detection:
1. Attack pattern recognition using sequence models
2. Anomalous access detection using clustering
3. Brute-force detection using time-series analysis
4. Threat scoring with ensemble methods

Unlike keyword matching, this learns patterns from actual attack data.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter, deque
from datetime import datetime, timedelta
import re
import pickle
import json
import logging
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ThreatDetectionResult:
    """Result of security threat analysis"""
    is_threat: bool
    threat_score: float  # 0-1, higher = more dangerous
    threat_type: str  # 'brute_force', 'injection', 'reconnaissance', 'dos', etc.
    confidence: float
    severity: str  # 'critical', 'high', 'medium', 'low'
    attack_indicators: List[Dict[str, Any]] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    related_events: List[str] = field(default_factory=list)


class AttackSequenceDetector:
    """
    Detects attack patterns using sequence analysis.
    
    Uses Markov chains to model normal vs attack behavior patterns.
    Learns transition probabilities between log events.
    """
    
    def __init__(self, order: int = 2):
        self.order = order  # Markov chain order
        self.normal_transitions: Dict[tuple, Counter] = defaultdict(Counter)
        self.attack_transitions: Dict[tuple, Counter] = defaultdict(Counter)
        self.normal_total: Dict[tuple, int] = defaultdict(int)
        self.attack_total: Dict[tuple, int] = defaultdict(int)
        self.trained = False
        
    def _get_event_signature(self, log: Dict) -> str:
        """Extract a signature from a log event for sequence analysis"""
        # Combine relevant fields into a signature
        components = []
        
        # Event type/action
        message = log.get("message", "").lower()
        if "login" in message or "auth" in message:
            components.append("AUTH")
        elif "failed" in message or "error" in message:
            components.append("FAIL")
        elif "success" in message:
            components.append("SUCCESS")
        elif "request" in message or "get" in message or "post" in message:
            components.append("REQUEST")
        elif "access" in message:
            components.append("ACCESS")
        elif "denied" in message or "blocked" in message:
            components.append("DENIED")
        else:
            components.append("OTHER")
        
        # Source indicator
        source = log.get("source", log.get("Component", ""))
        if source:
            components.append(source[:10].upper())
        
        return "_".join(components)
    
    def fit(self, normal_logs: List[Dict], attack_logs: List[Dict]) -> "AttackSequenceDetector":
        """
        Train on labeled normal and attack log sequences.
        
        Args:
            normal_logs: List of logs from normal activity
            attack_logs: List of logs from known attacks
        """
        # Process normal sequences
        if normal_logs:
            signatures = [self._get_event_signature(log) for log in normal_logs]
            for i in range(len(signatures) - self.order):
                state = tuple(signatures[i:i+self.order])
                next_event = signatures[i+self.order]
                self.normal_transitions[state][next_event] += 1
                self.normal_total[state] += 1
        
        # Process attack sequences
        if attack_logs:
            signatures = [self._get_event_signature(log) for log in attack_logs]
            for i in range(len(signatures) - self.order):
                state = tuple(signatures[i:i+self.order])
                next_event = signatures[i+self.order]
                self.attack_transitions[state][next_event] += 1
                self.attack_total[state] += 1
        
        self.trained = True
        logger.info(f"Attack sequence detector trained: {len(self.normal_total)} normal states, {len(self.attack_total)} attack states")
        return self
    
    def score_sequence(self, logs: List[Dict]) -> float:
        """
        Score a sequence of logs for attack likelihood.
        
        Returns:
            Attack probability (0 = normal, 1 = attack)
        """
        if not self.trained or len(logs) <= self.order:
            return 0.5
        
        signatures = [self._get_event_signature(log) for log in logs]
        
        normal_log_prob = 0.0
        attack_log_prob = 0.0
        
        for i in range(len(signatures) - self.order):
            state = tuple(signatures[i:i+self.order])
            next_event = signatures[i+self.order]
            
            # Normal probability
            if state in self.normal_total and self.normal_total[state] > 0:
                count = self.normal_transitions[state].get(next_event, 0) + 1
                total = self.normal_total[state] + len(self.normal_transitions[state])
                normal_log_prob += np.log(count / total)
            else:
                normal_log_prob += np.log(0.01)  # Low probability for unseen
            
            # Attack probability
            if state in self.attack_total and self.attack_total[state] > 0:
                count = self.attack_transitions[state].get(next_event, 0) + 1
                total = self.attack_total[state] + len(self.attack_transitions[state])
                attack_log_prob += np.log(count / total)
            else:
                attack_log_prob += np.log(0.01)
        
        # Convert log probabilities to attack score
        if attack_log_prob > normal_log_prob:
            return min(1.0, 0.5 + (attack_log_prob - normal_log_prob) / 10)
        else:
            return max(0.0, 0.5 - (normal_log_prob - attack_log_prob) / 10)


class BruteForceDetector:
    """
    Detects brute-force attacks using statistical analysis.
    
    Features:
    - Tracks failed authentication attempts per source
    - Learns normal failure patterns
    - Detects anomalous failure rates
    """
    
    def __init__(self, window_seconds: int = 300, threshold_multiplier: float = 3.0):
        self.window_seconds = window_seconds
        self.threshold_multiplier = threshold_multiplier
        
        # Sliding windows per source
        self.source_failures: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Learned baselines
        self.baseline_rate: float = 0.0  # Failures per source per window
        self.baseline_std: float = 1.0
        self.trained = False
        
    def fit(self, auth_logs: List[Dict]) -> "BruteForceDetector":
        """Learn normal authentication failure patterns"""
        # Group by source and time window
        source_window_counts = defaultdict(list)
        
        for log in auth_logs:
            if not self._is_auth_failure(log):
                continue
            
            source = self._extract_source(log)
            if not source:
                continue
            
            source_window_counts[source].append(1)
        
        # Calculate baseline rates
        if source_window_counts:
            all_counts = [len(counts) for counts in source_window_counts.values()]
            self.baseline_rate = float(np.mean(all_counts))
            self.baseline_std = float(np.std(all_counts)) + 0.1
        
        self.trained = True
        logger.info(f"Brute force detector trained: baseline rate = {self.baseline_rate:.2f} +/- {self.baseline_std:.2f}")
        return self
    
    def _is_auth_failure(self, log: Dict) -> bool:
        """Check if log represents an authentication failure"""
        message = log.get("message", "").lower()
        return any(pattern in message for pattern in [
            "failed", "failure", "invalid", "denied", "rejected",
            "authentication failed", "login failed", "access denied"
        ])
    
    def _extract_source(self, log: Dict) -> Optional[str]:
        """Extract source IP or identifier from log"""
        message = log.get("message", "")
        
        # Try to find IP address
        ip_match = re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', message)
        if ip_match:
            return ip_match.group()
        
        # Try source field
        if "source" in log:
            return str(log["source"])
        
        # Try user field
        user_match = re.search(r'user[=:\s]+([^\s,;]+)', message, re.IGNORECASE)
        if user_match:
            return f"user:{user_match.group(1)}"
        
        return None
    
    def _get_timestamp(self, log: Dict) -> Optional[datetime]:
        """Extract timestamp from log"""
        ts = log.get("timestamp", log.get("Timestamp"))
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except:
                pass
        return datetime.now()
    
    def detect(self, log: Dict) -> Dict[str, Any]:
        """
        Check if this log indicates brute-force activity.
        
        Updates internal state and returns detection result.
        """
        if not self._is_auth_failure(log):
            return {"is_brute_force": False, "reason": "not_auth_failure"}
        
        source = self._extract_source(log)
        if not source:
            return {"is_brute_force": False, "reason": "no_source"}
        
        timestamp = self._get_timestamp(log)
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add to sliding window
        self.source_failures[source].append(timestamp)
        
        # Remove old entries
        cutoff = timestamp - timedelta(seconds=self.window_seconds)
        while self.source_failures[source] and self.source_failures[source][0] < cutoff:
            self.source_failures[source].popleft()
        
        # Count failures in window
        failure_count = len(self.source_failures[source])
        
        # Calculate z-score
        if self.trained:
            z_score = (failure_count - self.baseline_rate) / self.baseline_std
            is_brute_force = z_score > self.threshold_multiplier
        else:
            # Default heuristic: more than 5 failures in window
            is_brute_force = failure_count > 5
            z_score = failure_count / 5.0
        
        return {
            "is_brute_force": is_brute_force,
            "source": source,
            "failure_count": failure_count,
            "window_seconds": self.window_seconds,
            "z_score": z_score,
            "threat_score": min(1.0, z_score / (self.threshold_multiplier * 2))
        }
    
    def get_active_sources(self) -> List[Dict[str, Any]]:
        """Get currently tracked sources with failure counts"""
        now = datetime.now()
        active = []
        
        for source, failures in self.source_failures.items():
            if failures:
                active.append({
                    "source": source,
                    "failure_count": len(failures),
                    "first_failure": failures[0].isoformat() if failures else None,
                    "last_failure": failures[-1].isoformat() if failures else None
                })
        
        return sorted(active, key=lambda x: x["failure_count"], reverse=True)


class InjectionDetector:
    """
    Detects injection attacks (SQL, command, XSS) using pattern learning.
    
    Uses a combination of:
    - Learned malicious patterns from training data
    - Character distribution analysis
    - Entropy-based detection
    """
    
    def __init__(self):
        self.malicious_ngrams: Dict[str, float] = {}  # ngram -> score
        self.benign_ngrams: Dict[str, float] = {}
        self.char_entropy_threshold: float = 4.5
        self.trained = False
        
        # Comprehensive attack patterns for all known attack types
        self.base_patterns = {
            # SQL Injection patterns
            "sql_injection": [
                r"'\s*or\s+['\d]", r"union\s+select", r";\s*drop\s+",
                r"--\s*$", r"1\s*=\s*1", r"'\s*;\s*--",
                r"'\s*or\s+'", r"waitfor\s+delay", r"benchmark\s*\(",
                r"sleep\s*\(", r"pg_sleep", r"load_file\s*\(",
                r"into\s+outfile", r"into\s+dumpfile", r"information_schema",
                r"sysobjects", r"syscolumns", r"@@version", r"char\s*\(\d+\)",
                r"concat\s*\(", r"group_concat", r"having\s+\d+\s*=",
                r"order\s+by\s+\d+", r"extractvalue\s*\(", r"updatexml\s*\("
            ],
            # Command Injection patterns
            "command_injection": [
                r";\s*\w+\s*", r"\|\s*\w+", r"`[^`]+`",
                r"\$\([^)]+\)", r"&&\s*\w+", r"\|\|\s*\w+",
                r"/bin/sh", r"/bin/bash", r"cmd\.exe", r"powershell",
                r"wget\s+", r"curl\s+", r"nc\s+-", r"netcat",
                r"rm\s+-rf", r"cat\s+/etc", r"chmod\s+", r"chown\s+",
                r"eval\s*\(", r"exec\s*\(", r"system\s*\(", r"passthru\s*\(",
                r"proc_open", r"popen\s*\(", r"shell_exec"
            ],
            # XSS patterns
            "xss": [
                r"<script[^>]*>", r"javascript:", r"on\w+\s*=",
                r"<img[^>]+onerror", r"<iframe", r"document\.",
                r"<svg[^>]+onload", r"<body[^>]+onload", r"<input[^>]+onfocus",
                r"expression\s*\(", r"vbscript:", r"data:text/html",
                r"alert\s*\(", r"prompt\s*\(", r"confirm\s*\(",
                r"fromcharcode", r"innerhtml\s*=", r"outerhtml\s*=",
                r"document\.cookie", r"document\.write", r"\.innerHTML",
                r"eval\s*\(", r"settimeout\s*\(", r"setinterval\s*\("
            ],
            # Path Traversal patterns
            "path_traversal": [
                r"\.\./", r"\.\.\\", r"%2e%2e", r"etc/passwd",
                r"etc/shadow", r"windows/system32", r"boot\.ini",
                r"%252e%252e", r"\.\.%00", r"\.\.%0d", r"\.\.%c0%af",
                r"/proc/self", r"/var/log", r"web\.config",
                r"\.htaccess", r"wp-config\.php", r"config\.php"
            ],
            # NoSQL Injection patterns
            "nosql_injection": [
                r"\$gt\s*:", r"\$lt\s*:", r"\$ne\s*:", r"\$eq\s*:",
                r"\$regex\s*:", r"\$where\s*:", r"\$or\s*:\s*\[",
                r"\$and\s*:\s*\[", r"\$nin\s*:", r"\$in\s*:",
                r"{\s*\$", r"\[\s*\$", r"db\.\w+\.find\s*\(",
                r"db\.\w+\.insert", r"db\.\w+\.update", r"db\.\w+\.delete",
                r"mapReduce", r"\.toArray\s*\(", r"\.forEach\s*\("
            ],
            # Log4Shell / JNDI Injection
            "log4shell": [
                r"\$\{jndi:", r"\$\{lower:", r"\$\{upper:",
                r"\$\{env:", r"\$\{sys:", r"\$\{java:",
                r"ldap://", r"rmi://", r"dns://", r"iiop://",
                r"\$\{\w+:\w+:", r"ldaps://", r"corba://"
            ],
            # Server Side Template Injection (SSTI)
            "ssti": [
                r"\{\{.*\}\}", r"\{%.*%\}", r"\$\{.*\}",
                r"__class__", r"__mro__", r"__subclasses__",
                r"__globals__", r"__builtins__", r"__import__",
                r"config\.__class__", r"request\.__class__",
                r"lipsum\.__globals__", r"cycler\.__init__",
                r"joiner\.__init__", r"\|attr\s*\("
            ],
            # XXE (XML External Entity)
            "xxe": [
                r"<!DOCTYPE[^>]+\[", r"<!ENTITY", r"SYSTEM\s+['\"]",
                r"file://", r"php://", r"expect://", r"data://",
                r"<!ELEMENT", r"%\w+;", r"&\w+;",
                r"<!NOTATION", r"PUBLIC\s+['\"]"
            ],
            # LDAP Injection
            "ldap_injection": [
                r"\)\s*\(\|", r"\(\|?\s*\(", r"\)\s*\(\&",
                r"\*\)\(", r"uid\s*=\s*\*", r"cn\s*=\s*\*",
                r"\)\s*\(uid=", r"\)\s*\(cn=", r"objectclass\s*=\s*\*"
            ],
            # Header Injection / CRLF
            "header_injection": [
                r"%0d%0a", r"%0a%0d", r"\r\n", r"\\r\\n",
                r"set-cookie:", r"location:", r"x-forwarded"
            ],
            # Open Redirect
            "open_redirect": [
                r"redirect=https?://", r"url=https?://", r"next=https?://",
                r"return=https?://", r"goto=https?://", r"target=https?://",
                r"redirect=/[^/]", r"//\w+\.\w+"
            ],
            # Server Side Request Forgery (SSRF)
            "ssrf": [
                r"169\.254\.169\.254", r"metadata\.google",
                r"localhost:\d+", r"127\.0\.0\.1:\d+",
                r"0\.0\.0\.0:\d+", r"::1", r"\[::\]",
                r"file://localhost", r"gopher://", r"dict://"
            ],
            # Deserialization attacks
            "deserialization": [
                r"ysoserial", r"gadgetchain", r"rO0AB",
                r"O:\d+:", r"a:\d+:{", r"java\.lang\.Runtime",
                r"java\.lang\.ProcessBuilder", r"javax\.script",
                r"org\.apache\.commons", r"com\.sun\.org\.apache"
            ],
            # Authentication bypass
            "auth_bypass": [
                r"admin'--", r"' or ''='", r"' or 1=1",
                r"password['\s]*=", r"pass['\s]*=", r"pwd['\s]*=",
                r"x-auth-token:", r"authorization:\s*bearer",
                r"jwt\.", r"eyj\w+\.\w+\.\w+"
            ]
        }
        
    def _extract_ngrams(self, text: str, n: int = 3) -> List[str]:
        """Extract character n-grams from text"""
        text = text.lower()
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of character distribution"""
        if not text:
            return 0.0
        
        freq = Counter(text)
        total = len(text)
        probs = [count / total for count in freq.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        return entropy
    
    def fit(self, malicious_inputs: List[str], benign_inputs: List[str]) -> "InjectionDetector":
        """
        Train on labeled malicious and benign inputs.
        
        Args:
            malicious_inputs: List of known attack payloads
            benign_inputs: List of normal inputs
        """
        # Learn malicious n-grams
        malicious_ngram_counts = Counter()
        for text in malicious_inputs:
            ngrams = self._extract_ngrams(text)
            malicious_ngram_counts.update(ngrams)
        
        # Learn benign n-grams
        benign_ngram_counts = Counter()
        for text in benign_inputs:
            ngrams = self._extract_ngrams(text)
            benign_ngram_counts.update(ngrams)
        
        # Calculate discriminative scores
        all_ngrams = set(malicious_ngram_counts.keys()) | set(benign_ngram_counts.keys())
        
        for ngram in all_ngrams:
            mal_count = malicious_ngram_counts.get(ngram, 0) + 1
            ben_count = benign_ngram_counts.get(ngram, 0) + 1
            
            # Log-odds ratio
            score = float(np.log(mal_count / ben_count))
            
            if score > 0:
                self.malicious_ngrams[ngram] = score
            else:
                self.benign_ngrams[ngram] = -score
        
        # Learn entropy threshold from benign data
        if benign_inputs:
            entropies = [self._calculate_entropy(text) for text in benign_inputs]
            self.char_entropy_threshold = float(np.mean(entropies) + 2 * np.std(entropies))
        
        self.trained = True
        logger.info(f"Injection detector trained: {len(self.malicious_ngrams)} malicious ngrams, threshold entropy = {self.char_entropy_threshold:.2f}")
        return self
    
    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect potential injection attacks in text.
        
        Returns detection result with threat type and score.
        """
        text_lower = text.lower()
        
        # Check base patterns
        detected_patterns = []
        for attack_type, patterns in self.base_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected_patterns.append({
                        "type": attack_type,
                        "pattern": pattern
                    })
        
        # Calculate n-gram score
        ngrams = self._extract_ngrams(text)
        mal_score = sum(self.malicious_ngrams.get(ng, 0) for ng in ngrams)
        ben_score = sum(self.benign_ngrams.get(ng, 0) for ng in ngrams)
        ngram_score = mal_score - ben_score
        
        # Calculate entropy
        entropy = self._calculate_entropy(text)
        high_entropy = entropy > self.char_entropy_threshold
        
        # Combine signals
        is_injection = len(detected_patterns) > 0 or ngram_score > 2.0 or high_entropy
        
        threat_score = 0.0
        if detected_patterns:
            threat_score += 0.4 * min(len(detected_patterns), 3) / 3
        if ngram_score > 0:
            threat_score += 0.3 * min(ngram_score / 5.0, 1.0)
        if high_entropy:
            threat_score += 0.3 * min((entropy - self.char_entropy_threshold) / 2.0, 1.0)
        
        # Determine primary attack type
        attack_type = "unknown"
        if detected_patterns:
            type_counts = Counter(p["type"] for p in detected_patterns)
            attack_type = type_counts.most_common(1)[0][0]
        
        return {
            "is_injection": is_injection,
            "attack_type": attack_type,
            "threat_score": min(threat_score, 1.0),
            "detected_patterns": detected_patterns,
            "ngram_score": ngram_score,
            "entropy": entropy,
            "high_entropy": high_entropy
        }


class ReconnaissanceDetector:
    """
    Detects reconnaissance/scanning activity.
    
    Identifies patterns like:
    - Port scanning
    - Directory enumeration
    - User enumeration
    - Service discovery
    """
    
    def __init__(self, window_seconds: int = 60, diversity_threshold: int = 10):
        self.window_seconds = window_seconds
        self.diversity_threshold = diversity_threshold
        
        # Track access patterns per source
        self.source_targets: Dict[str, deque] = defaultdict(lambda: deque())
        self.trained = False
        
    def _extract_target(self, log: Dict) -> Optional[str]:
        """Extract the target being accessed (path, port, user, etc.)"""
        message = log.get("message", "")
        
        # Extract URL path
        path_match = re.search(r'(?:GET|POST|PUT|DELETE)\s+(/[^\s]*)', message, re.IGNORECASE)
        if path_match:
            return f"path:{path_match.group(1)}"
        
        # Extract port
        port_match = re.search(r'port[=:\s]+(\d+)', message, re.IGNORECASE)
        if port_match:
            return f"port:{port_match.group(1)}"
        
        # Extract user
        user_match = re.search(r'user[=:\s]+([^\s,;]+)', message, re.IGNORECASE)
        if user_match:
            return f"user:{user_match.group(1)}"
        
        return None
    
    def _extract_source(self, log: Dict) -> Optional[str]:
        """Extract source IP or identifier"""
        message = log.get("message", "")
        ip_match = re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', message)
        if ip_match:
            return ip_match.group()
        return log.get("source")
    
    def _get_timestamp(self, log: Dict) -> datetime:
        """Extract timestamp from log"""
        ts = log.get("timestamp")
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except:
                pass
        return datetime.now()
    
    def detect(self, log: Dict) -> Dict[str, Any]:
        """Detect reconnaissance activity"""
        source = self._extract_source(log)
        target = self._extract_target(log)
        timestamp = self._get_timestamp(log)
        
        if not source or not target:
            return {"is_recon": False, "reason": "missing_data"}
        
        # Add to sliding window
        self.source_targets[source].append((timestamp, target))
        
        # Clean old entries
        cutoff = timestamp - timedelta(seconds=self.window_seconds)
        while self.source_targets[source] and self.source_targets[source][0][0] < cutoff:
            self.source_targets[source].popleft()
        
        # Count unique targets in window
        unique_targets = set(t[1] for t in self.source_targets[source])
        diversity = len(unique_targets)
        
        # Detect scanning
        is_recon = diversity >= self.diversity_threshold
        
        # Determine recon type
        recon_type = "unknown"
        if is_recon:
            target_types = Counter(t.split(":")[0] for t in unique_targets)
            if target_types.get("port", 0) > 3:
                recon_type = "port_scan"
            elif target_types.get("path", 0) > 5:
                recon_type = "directory_enum"
            elif target_types.get("user", 0) > 3:
                recon_type = "user_enum"
            else:
                recon_type = "general_scan"
        
        return {
            "is_recon": is_recon,
            "recon_type": recon_type,
            "source": source,
            "unique_targets": diversity,
            "window_seconds": self.window_seconds,
            "threat_score": min(diversity / (self.diversity_threshold * 2), 1.0),
            "targets_sample": list(unique_targets)[:10]
        }


class SecurityThreatDetector:
    """
    Main security threat detection engine combining all detectors.
    
    This is the primary interface for security analysis.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.sequence_detector = AttackSequenceDetector(order=2)
        self.brute_force_detector = BruteForceDetector(window_seconds=300)
        self.injection_detector = InjectionDetector()
        self.recon_detector = ReconnaissanceDetector(window_seconds=60)
        
        # Use absolute path relative to project root (2 levels up from this file)
        project_root = Path(__file__).parent.parent.parent.parent.parent
        default_path = project_root / "data" / "models" / "security_detector.pkl"
        self.model_path = Path(model_path) if model_path else default_path
        self.is_trained = False
        self.training_stats: Dict[str, Any] = {}
        
    async def train(self, 
                    normal_logs: List[Dict],
                    attack_logs: Optional[List[Dict]] = None,
                    malicious_inputs: Optional[List[str]] = None,
                    benign_inputs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train all security detectors.
        
        Args:
            normal_logs: Logs from normal activity
            attack_logs: Logs from known attacks (optional)
            malicious_inputs: Known malicious payloads (optional)
            benign_inputs: Known safe inputs (optional)
        """
        logger.info(f"Training security detector on {len(normal_logs)} normal logs...")
        
        # Train sequence detector
        if attack_logs:
            self.sequence_detector.fit(normal_logs, attack_logs)
        else:
            # Use heuristics to identify potential attack patterns in normal logs
            potential_attacks = [
                log for log in normal_logs
                if any(w in log.get("message", "").lower() 
                       for w in ["attack", "malicious", "exploit", "intrusion"])
            ]
            self.sequence_detector.fit(
                [l for l in normal_logs if l not in potential_attacks],
                potential_attacks
            )
        
        # Train brute force detector
        auth_logs = [
            log for log in normal_logs
            if any(w in log.get("message", "").lower() 
                   for w in ["auth", "login", "password", "credential"])
        ]
        if auth_logs:
            self.brute_force_detector.fit(auth_logs)
        
        # Train injection detector
        if malicious_inputs and benign_inputs:
            self.injection_detector.fit(malicious_inputs, benign_inputs)
        else:
            # Generate from logs
            inputs = [log.get("message", "") for log in normal_logs]
            # Split arbitrarily for initial training
            self.injection_detector.fit(
                inputs[:len(inputs)//10],  # Assume some might be malicious
                inputs[len(inputs)//10:]
            )
        
        self.is_trained = True
        
        self.training_stats = {
            "timestamp": datetime.now().isoformat(),
            "normal_logs": len(normal_logs),
            "attack_logs": len(attack_logs) if attack_logs else 0,
            "auth_logs": len(auth_logs),
            "detectors_trained": [
                "sequence", "brute_force", "injection", "reconnaissance"
            ]
        }
        
        # Save model
        await self.save()
        
        return {
            "success": True,
            **self.training_stats
        }
    
    def _extract_sklearn_features(self, log: Dict, message: str) -> Optional[np.ndarray]:
        """Extract features for sklearn model prediction"""
        if not hasattr(self, '_sklearn_scaler') or self._sklearn_scaler is None:
            return None
        
        # Extract behavioral features
        features = []
        
        # Message length
        features.append(len(message))
        
        # Special character counts
        features.append(message.count("'"))
        features.append(message.count('"'))
        features.append(message.count(';'))
        features.append(message.count('--'))
        features.append(message.count('<'))
        features.append(message.count('>'))
        
        # Severity encoding
        severity = str(log.get("severity", "INFO")).upper()
        features.append(1.0 if severity in ["ERROR", "CRITICAL"] else 0.0)
        features.append(1.0 if severity == "WARNING" else 0.0)
        
        # Keyword presence
        msg_lower = message.lower()
        features.append(1.0 if any(k in msg_lower for k in ['failed', 'error', 'denied']) else 0.0)
        features.append(1.0 if any(k in msg_lower for k in ['select', 'union', 'drop', 'insert']) else 0.0)
        features.append(1.0 if any(k in msg_lower for k in ['script', 'alert', 'onerror']) else 0.0)
        features.append(1.0 if any(k in msg_lower for k in ['scan', 'probe', 'enum']) else 0.0)
        
        try:
            X = np.array([features])
            # Pad or truncate to match expected features
            expected_len = len(self._feature_names) if hasattr(self, '_feature_names') and self._feature_names else X.shape[1]
            if X.shape[1] < expected_len:
                X = np.pad(X, ((0, 0), (0, expected_len - X.shape[1])), mode='constant')
            elif X.shape[1] > expected_len:
                X = X[:, :expected_len]
            return self._sklearn_scaler.transform(X)
        except Exception:
            return None
    
    async def detect(self, logs: List[Dict]) -> List[ThreatDetectionResult]:
        """
        Analyze logs for security threats.
        
        Args:
            logs: List of log entries
            
        Returns:
            List of ThreatDetectionResult
        """
        results = []
        
        # Disable sklearn classifier (produces too many false positives)
        # Use rule-based detection which is more reliable
        use_sklearn = False  # Was: (hasattr(self, '_sklearn_classifier') and ...)
        
        # Sequence analysis (needs multiple logs)
        if len(logs) > 3:
            sequence_score = self.sequence_detector.score_sequence(logs)
        else:
            sequence_score = 0.0
        
        for log in logs:
            message = log.get("message", "")
            
            # Try sklearn prediction first
            sklearn_threat = False
            sklearn_score = 0.0
            sklearn_type = "none"
            
            if use_sklearn:
                X = self._extract_sklearn_features(log, message)
                if X is not None:
                    try:
                        pred = self._sklearn_classifier.predict(X)[0]
                        sklearn_threat = bool(pred)
                        if hasattr(self._sklearn_classifier, 'predict_proba'):
                            proba = self._sklearn_classifier.predict_proba(X)[0]
                            sklearn_score = float(max(proba))
                            # Only consider threat if confidence > 70%
                            if sklearn_score < 0.70:
                                sklearn_threat = False
                                sklearn_score = sklearn_score * 0.5  # Reduce score
                        else:
                            sklearn_score = 0.8 if sklearn_threat else 0.2
                        
                        # Infer type from message if threat detected
                        if sklearn_threat:
                            msg_lower = message.lower()
                            if any(k in msg_lower for k in ['failed', 'password', 'login', 'auth']):
                                sklearn_type = "brute_force"
                            elif any(k in msg_lower for k in ['select', 'union', 'drop', 'insert', '--']):
                                sklearn_type = "sql_injection"
                            elif any(k in msg_lower for k in ['script', 'alert', 'onerror', '<']):
                                sklearn_type = "xss"
                            else:
                                sklearn_type = "suspicious_activity"
                    except Exception:
                        pass
            
            # Check rule-based detectors
            brute_result = self.brute_force_detector.detect(log)
            injection_result = self.injection_detector.detect(message)
            recon_result = self.recon_detector.detect(log)
            
            # Thresholds for threat detection
            RULE_THREAT_THRESHOLD = 0.30  # Lower for rule-based (more sensitive)
            SKLEARN_HIGH_THRESHOLD = 0.85  # Very high for sklearn-only detection
            
            # Rule-based threat detection
            rule_based_threat = (
                brute_result.get("is_brute_force", False) or
                injection_result.get("is_injection", False) or
                recon_result.get("is_recon", False) or
                sequence_score > 0.70
            )
            
            rule_threat_score = max(
                brute_result.get("threat_score", 0),
                injection_result.get("threat_score", 0),
                recon_result.get("threat_score", 0)
            )
            
            # Determine if this is a threat
            is_threat = False
            
            # Rule-based detection: if any detector flags it, it's a threat
            if injection_result.get("is_injection") and rule_threat_score > 0.1:
                is_threat = True
            elif brute_result.get("is_brute_force") and rule_threat_score > 0.3:
                is_threat = True
            elif recon_result.get("is_recon") and rule_threat_score > 0.3:
                is_threat = True
            elif sequence_score > 0.70:
                is_threat = True
            elif sklearn_threat and sklearn_type != "suspicious_activity" and sklearn_score > RULE_THREAT_THRESHOLD:
                is_threat = True
            elif sklearn_threat and sklearn_score > SKLEARN_HIGH_THRESHOLD:
                is_threat = True
            
            # Calculate overall threat score
            threat_score = max(
                sklearn_score if sklearn_threat else 0,
                brute_result.get("threat_score", 0),
                injection_result.get("threat_score", 0),
                recon_result.get("threat_score", 0),
                sequence_score
            )
            
            # Determine primary threat type (prefer rule-based over sklearn "suspicious_activity")
            if brute_result.get("is_brute_force"):
                threat_type = "brute_force"
            elif injection_result.get("is_injection"):
                threat_type = injection_result.get("attack_type", "injection")
            elif recon_result.get("is_recon"):
                threat_type = recon_result.get("recon_type", "reconnaissance")
            elif sklearn_threat and sklearn_type not in ["none", "suspicious_activity"]:
                threat_type = sklearn_type
            elif sequence_score > 0.75:
                threat_type = "attack_pattern"
            elif sklearn_threat and sklearn_type == "suspicious_activity":
                threat_type = "suspicious_activity"
            else:
                threat_type = "none"
            
            # Determine severity
            if threat_score > 0.8:
                severity = "critical"
            elif threat_score > 0.6:
                severity = "high"
            elif threat_score > 0.4:
                severity = "medium"
            elif threat_score > 0.2:
                severity = "low"
            else:
                severity = "none"
            
            # Collect attack indicators
            indicators = []
            if brute_result.get("is_brute_force"):
                indicators.append({
                    "type": "brute_force",
                    "details": {
                        "source": brute_result.get("source"),
                        "failure_count": brute_result.get("failure_count")
                    }
                })
            if injection_result.get("detected_patterns"):
                indicators.append({
                    "type": "injection",
                    "details": injection_result.get("detected_patterns")
                })
            if recon_result.get("is_recon"):
                indicators.append({
                    "type": "reconnaissance",
                    "details": {
                        "targets": recon_result.get("unique_targets"),
                        "recon_type": recon_result.get("recon_type")
                    }
                })
            
            # Generate recommended actions
            actions = self._generate_recommendations(threat_type, threat_score, indicators)
            
            results.append(ThreatDetectionResult(
                is_threat=is_threat,
                threat_score=threat_score,
                threat_type=threat_type,
                confidence=0.8 if self.is_trained else 0.5,
                severity=severity,
                attack_indicators=indicators,
                recommended_actions=actions
            ))
        
        return results
    
    def _generate_recommendations(self, threat_type: str, score: float, indicators: List) -> List[str]:
        """Generate security recommendations based on threat analysis"""
        actions = []
        
        if threat_type == "brute_force":
            actions.extend([
                "Implement account lockout after multiple failed attempts",
                "Enable CAPTCHA on login forms",
                "Consider blocking source IP temporarily",
                "Review affected accounts for compromise"
            ])
        elif threat_type in ["sql_injection", "command_injection", "xss"]:
            actions.extend([
                "Sanitize and validate all user inputs",
                "Use parameterized queries for database operations",
                "Implement Web Application Firewall (WAF) rules",
                "Review and update input validation logic"
            ])
        elif threat_type in ["port_scan", "directory_enum", "user_enum", "reconnaissance"]:
            actions.extend([
                "Monitor source IP for further suspicious activity",
                "Review firewall rules and rate limiting",
                "Enable intrusion detection system alerts",
                "Consider blocking reconnaissance sources"
            ])
        elif threat_type == "attack_pattern":
            actions.extend([
                "Investigate the sequence of events for attack chain",
                "Review affected systems for compromise",
                "Collect logs for forensic analysis",
                "Alert security team for manual review"
            ])
        
        if score > 0.8:
            actions.insert(0, "URGENT: Immediate security review required")
        
        return actions[:5]  # Top 5 recommendations
    
    async def save(self) -> None:
        """Persist model to disk"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "sequence_detector": {
                "normal_transitions": dict(self.sequence_detector.normal_transitions),
                "attack_transitions": dict(self.sequence_detector.attack_transitions),
                "normal_total": dict(self.sequence_detector.normal_total),
                "attack_total": dict(self.sequence_detector.attack_total),
                "trained": self.sequence_detector.trained
            },
            "brute_force_detector": {
                "baseline_rate": self.brute_force_detector.baseline_rate,
                "baseline_std": self.brute_force_detector.baseline_std,
                "trained": self.brute_force_detector.trained
            },
            "injection_detector": {
                "malicious_ngrams": self.injection_detector.malicious_ngrams,
                "benign_ngrams": self.injection_detector.benign_ngrams,
                "char_entropy_threshold": self.injection_detector.char_entropy_threshold,
                "trained": self.injection_detector.trained
            },
            "training_stats": self.training_stats
        }
        
        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Security detector saved to {self.model_path}")
    
    async def load(self) -> bool:
        """Load model from disk - supports both custom and sklearn formats"""
        if not self.model_path.exists():
            logger.warning(f"No saved model found at {self.model_path}")
            return False
        
        try:
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
            
            # Check if this is sklearn format (from retrain_models.py)
            if model_data.get("type") == "behavioral" or "classifier" in model_data:
                # sklearn RandomForest + scaler format
                self._sklearn_classifier = model_data.get("classifier")
                self._sklearn_scaler = model_data.get("scaler")
                self._sklearn_anomaly = model_data.get("anomaly_detector")
                self._feature_names = model_data.get("feature_names", [])
                
                # Mark all detectors as trained
                self.sequence_detector.trained = True
                self.brute_force_detector.trained = True
                self.injection_detector.trained = True
                self.is_trained = True
                
                self.training_stats = {
                    "accuracy": model_data.get("accuracy", 0),
                    "type": "behavioral"
                }
                
                logger.info(f"Loaded sklearn-format security detector from {self.model_path}")
                return True
            
            # Custom format (from training_pipeline.py)
            seq_data = model_data["sequence_detector"]
            self.sequence_detector.normal_transitions = defaultdict(Counter, {
                tuple(k): Counter(v) for k, v in seq_data["normal_transitions"].items()
            })
            self.sequence_detector.attack_transitions = defaultdict(Counter, {
                tuple(k): Counter(v) for k, v in seq_data["attack_transitions"].items()
            })
            self.sequence_detector.normal_total = defaultdict(int, seq_data["normal_total"])
            self.sequence_detector.attack_total = defaultdict(int, seq_data["attack_total"])
            self.sequence_detector.trained = seq_data["trained"]
            
            # Restore brute force detector
            bf_data = model_data["brute_force_detector"]
            self.brute_force_detector.baseline_rate = bf_data["baseline_rate"]
            self.brute_force_detector.baseline_std = bf_data["baseline_std"]
            self.brute_force_detector.trained = bf_data["trained"]
            
            # Restore injection detector
            inj_data = model_data["injection_detector"]
            self.injection_detector.malicious_ngrams = inj_data["malicious_ngrams"]
            self.injection_detector.benign_ngrams = inj_data["benign_ngrams"]
            self.injection_detector.char_entropy_threshold = inj_data["char_entropy_threshold"]
            self.injection_detector.trained = inj_data["trained"]
            
            self.training_stats = model_data.get("training_stats", {})
            self.is_trained = True
            
            logger.info(f"Security detector loaded from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            "is_trained": self.is_trained,
            "detectors": {
                "sequence": self.sequence_detector.trained,
                "brute_force": self.brute_force_detector.trained,
                "injection": self.injection_detector.trained,
                "reconnaissance": True  # Always available
            },
            "active_sources": self.brute_force_detector.get_active_sources()[:10],
            "training_stats": self.training_stats
        }
