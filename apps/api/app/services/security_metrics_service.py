"""
Security Metrics Service - Analyzes logs for security patterns.

Separated from metrics_service for better maintainability.
Uses configurable patterns from security_patterns.json.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import settings
from ..core.logging import get_logger
from ..storage.db import get_db, release_db

logger = get_logger(__name__)

# Load patterns on module import
_PATTERNS_PATH = Path(__file__).parent / "security_patterns.json"
_PATTERNS_DATA: Optional[Dict] = None


def _load_patterns() -> Dict[str, Any]:
    """Load security patterns from JSON file."""
    global _PATTERNS_DATA
    if _PATTERNS_DATA is None:
        try:
            with open(_PATTERNS_PATH, 'r') as f:
                _PATTERNS_DATA = json.load(f)
            logger.info(f"Loaded {len(_PATTERNS_DATA.get('patterns', {}))} security pattern categories")
        except Exception as e:
            logger.error(f"Failed to load security patterns: {e}")
            _PATTERNS_DATA = {"patterns": {}, "scoring": {}, "risk_levels": {}}
    return _PATTERNS_DATA


class SecurityMetricsService:
    """Service for computing security-related metrics."""
    
    def __init__(self):
        self.config = _load_patterns()
        self.patterns = self.config.get("patterns", {})
        self.scoring = self.config.get("scoring", {})
        self.risk_levels = self.config.get("risk_levels", {})
    
    async def get_security_metrics(
        self,
        tenant_id: Optional[str] = None,
        service_name: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze logs for security patterns.
        
        Uses stratified sampling from high-severity logs first,
        then samples from each service proportionally.
        """
        if tenant_id is None:
            tenant_id = settings.tenant_id_default
        
        db = await get_db()
        
        try:
            return await self._analyze_logs(
                db=db,
                tenant_id=tenant_id,
                service_name=service_name,
                from_time=from_time,
                to_time=to_time,
            )
        except Exception as e:
            logger.error(f"Security metrics error: {e}", exc_info=True)
            return self._empty_result(str(e))
        finally:
            await release_db(db)
    
    async def _analyze_logs(
        self,
        db,
        tenant_id: str,
        service_name: Optional[str],
        from_time: Optional[str],
        to_time: Optional[str],
    ) -> Dict[str, Any]:
        """Core analysis logic."""
        
        # Build query parameters
        params: List[Any] = [tenant_id]
        filters = ["tenant_id = ?"]
        
        if service_name:
            filters.append("service_name = ?")
            params.append(service_name)
        
        if from_time:
            filters.append("timestamp_utc >= ?")
            params.append(from_time)
        
        if to_time:
            filters.append("timestamp_utc <= ?")
            params.append(to_time)
        
        where_clause = " AND ".join(filters)
        
        # Strategy: Sample from high-severity logs first (they're more security-relevant)
        # Then fill in with samples from other logs
        all_rows = []
        
        # 1. Get high-severity logs first (severity >= 3 = warning and above)
        cursor = await db.execute(f"""
            SELECT body_raw, timestamp_utc, service_name, severity
            FROM logs_stream
            WHERE {where_clause} AND severity >= 3
            ORDER BY severity DESC, timestamp_utc DESC
            LIMIT 5000
        """, params)
        high_severity_rows = await cursor.fetchall()
        all_rows.extend(high_severity_rows)
        
        # 2. Get samples from each service (for better coverage)
        cursor = await db.execute(f"""
            SELECT DISTINCT service_name 
            FROM logs_stream 
            WHERE {where_clause}
        """, params)
        services = await cursor.fetchall()
        
        # Sample proportionally from remaining services
        samples_per_service = max(500, 5000 // max(len(services), 1))
        
        for svc_row in services[:20]:  # Max 20 services
            svc_name = svc_row["service_name"]
            svc_params = params.copy()
            
            # Skip if we're already filtering by service
            if service_name and svc_name != service_name:
                continue
            
            # Add service filter if not already filtered
            if not service_name:
                svc_params.append(svc_name)
                svc_where = f"{where_clause} AND service_name = ?"
            else:
                svc_where = where_clause
            
            cursor = await db.execute(f"""
                SELECT body_raw, timestamp_utc, service_name, severity
                FROM logs_stream
                WHERE {svc_where} AND severity < 3
                ORDER BY timestamp_utc DESC
                LIMIT ?
            """, svc_params + [samples_per_service])
            
            svc_rows = await cursor.fetchall()
            all_rows.extend(svc_rows)
        
        # Analyze all collected rows
        results = self._analyze_rows(all_rows)
        
        # Get affected services
        cursor = await db.execute(f"""
            SELECT service_name, COUNT(*) as count
            FROM logs_stream
            WHERE {where_clause} AND severity >= 4
            GROUP BY service_name
            ORDER BY count DESC
            LIMIT 10
        """, params)
        affected_rows = await cursor.fetchall()
        
        results["top_affected_services"] = [
            {"service": r["service_name"], "error_count": r["count"]}
            for r in affected_rows
        ]
        
        return results
    
    def _analyze_rows(self, rows: list) -> Dict[str, Any]:
        """Analyze log rows for security patterns."""
        security_counts: Dict[str, int] = {cat: 0 for cat in self.patterns.keys()}
        security_examples: Dict[str, List[Dict]] = {cat: [] for cat in self.patterns.keys()}
        matched_log_ids: set = set()
        
        for row in rows:
            body = (row["body_raw"] or "").lower()
            row_matched = False
            
            for category, pattern_info in self.patterns.items():
                keywords = pattern_info.get("keywords", [])
                
                for keyword in keywords:
                    if keyword.lower() in body:
                        security_counts[category] += 1
                        row_matched = True
                        
                        # Store example (max 5 per category)
                        if len(security_examples[category]) < 5:
                            security_examples[category].append({
                                "message": (row["body_raw"] or "")[:300],
                                "timestamp": row["timestamp_utc"],
                                "service": row["service_name"],
                                "severity": row["severity"],
                                "matched_keyword": keyword
                            })
                        break  # Count each log once per category
            
            if row_matched:
                matched_log_ids.add(id(row))  # Just for counting unique matches
        
        # Calculate security score
        total_logs = len(rows) if rows else 1
        total_security_events = sum(security_counts.values())
        
        security_score = self._calculate_score(security_counts, total_logs)
        risk_level = self._determine_risk_level(security_score, security_counts)
        
        return {
            "security_score": round(security_score, 1),
            "risk_level": risk_level,
            "total_security_events": total_security_events,
            "total_logs_analyzed": len(rows),
            "categories": security_counts,
            "examples": security_examples,
            "recommendations": self._generate_recommendations(security_counts, risk_level),
            "top_affected_services": [],  # Filled in by caller
        }
    
    def _calculate_score(self, counts: Dict[str, int], total_logs: int) -> float:
        """Calculate security score based on issue counts."""
        base_score = self.scoring.get("base_score", 100)
        deduction_per_issue = self.scoring.get("deduction_per_issue", 0.1)
        max_deduction = self.scoring.get("max_deduction", 100)
        
        # Weight issues by severity boost
        weighted_issues = 0
        for category, count in counts.items():
            boost = self.patterns.get(category, {}).get("severity_boost", 1)
            weighted_issues += count * boost
        
        # Calculate deduction
        issue_ratio = weighted_issues / total_logs if total_logs > 0 else 0
        deduction = min(issue_ratio * 100, max_deduction)
        
        return max(0, base_score - deduction)
    
    def _determine_risk_level(self, score: float, counts: Dict[str, int]) -> str:
        """Determine risk level based on score and critical thresholds."""
        critical = self.scoring.get("critical_threshold", {})
        
        # Check for critical thresholds first
        if counts.get("suspicious_access", 0) > critical.get("suspicious_access", 10):
            return "CRITICAL"
        if counts.get("brute_force_indicators", 0) > critical.get("brute_force_indicators", 20):
            return "CRITICAL"
        
        # Use score-based levels
        for level, config in sorted(
            self.risk_levels.items(),
            key=lambda x: x[1].get("min_score", 0),
            reverse=True
        ):
            if score >= config.get("min_score", 0):
                return level
        
        return "CRITICAL"
    
    def _generate_recommendations(self, counts: Dict[str, int], risk_level: str) -> List[str]:
        """Generate actionable security recommendations."""
        recommendations = []
        
        if counts.get("authentication_failures", 0) > 10:
            recommendations.append(
                "🔐 High authentication failures detected. Review password policies, "
                "consider implementing MFA, and check for compromised credentials."
            )
        
        if counts.get("brute_force_indicators", 0) > 5:
            recommendations.append(
                "⚠️ Brute force indicators found. Implement account lockout policies, "
                "add CAPTCHA, and consider IP-based rate limiting."
            )
        
        if counts.get("suspicious_access", 0) > 0:
            recommendations.append(
                "🚨 Suspicious access patterns detected. Review WAF rules, "
                "check for known vulnerability exploits, and audit access logs."
            )
        
        if counts.get("network_issues", 0) > 50:
            recommendations.append(
                "🌐 Elevated network issues detected. Check for DDoS patterns, "
                "review firewall rules, and monitor connection pools."
            )
        
        if counts.get("system_errors", 0) > 100:
            recommendations.append(
                "💥 High system error rate. Review application logs, "
                "check resource utilization, and investigate crash patterns."
            )
        
        if risk_level == "CRITICAL":
            recommendations.insert(0, 
                "🔴 CRITICAL: Immediate security review recommended. "
                "Consider incident response procedures."
            )
        
        if not recommendations:
            recommendations.append("✅ No significant security concerns detected.")
        
        return recommendations
    
    def _empty_result(self, error: str) -> Dict[str, Any]:
        """Return empty result structure on error."""
        return {
            "security_score": 100,
            "risk_level": "LOW",
            "total_security_events": 0,
            "total_logs_analyzed": 0,
            "categories": {cat: 0 for cat in self.patterns.keys()},
            "examples": {},
            "top_affected_services": [],
            "recommendations": [f"⚠️ Unable to analyze: {error}"],
            "error": error
        }


# Singleton instance
_security_service: Optional[SecurityMetricsService] = None


def get_security_metrics_service() -> SecurityMetricsService:
    """Get singleton security metrics service."""
    global _security_service
    if _security_service is None:
        _security_service = SecurityMetricsService()
    return _security_service
