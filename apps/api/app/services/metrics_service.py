"""
Metrics service - generates performance and security metrics.
Optimized for fast response times with large datasets.
"""

from typing import List, Optional, Dict, Any

from ..core.config import settings
from ..core.logging import get_logger
from ..storage.db import get_db

logger = get_logger(__name__)

# Limit queries to avoid timeouts on large datasets
MAX_SCAN_ROWS = 50000


class MetricsService:
    """Service for computing performance and security metrics."""
    
    async def get_performance_metrics(
        self,
        tenant_id: Optional[str] = None,
        service_name: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get performance metrics using optimized queries."""
        if tenant_id is None:
            tenant_id = settings.tenant_id_default
            
        db = await get_db()
        
        # Build filters
        service_filter = ""
        params: List[Any] = [tenant_id]
        if service_name:
            service_filter = "AND service_name = ?"
            params.append(service_name)
        
        time_filter = ""
        if from_time:
            time_filter += " AND timestamp_utc >= ?"
            params.append(from_time)
        if to_time:
            time_filter += " AND timestamp_utc <= ?"
            params.append(to_time)
        
        try:
            # Fast count with LIMIT
            cursor = await db.execute(f"""
                SELECT COUNT(*) as total FROM (
                    SELECT 1 FROM logs_stream 
                    WHERE tenant_id = ? {service_filter} {time_filter}
                    LIMIT {MAX_SCAN_ROWS}
                )
            """, params)
            row = await cursor.fetchone()
            total_logs = row["total"] if row else 0
            is_estimated = total_logs >= MAX_SCAN_ROWS
            
            # Severity breakdown - sampled
            cursor = await db.execute(f"""
                SELECT severity, COUNT(*) as count 
                FROM (
                    SELECT severity FROM logs_stream 
                    WHERE tenant_id = ? {service_filter} {time_filter}
                    LIMIT {MAX_SCAN_ROWS}
                )
                GROUP BY severity
            """, params)
            rows = await cursor.fetchall()
            severity_breakdown = {row["severity"]: row["count"] for row in rows}
            
            # Error rate
            error_count = sum(count for sev, count in severity_breakdown.items() if sev >= 4)
            warn_count = severity_breakdown.get(3, 0)
            error_rate = (error_count / total_logs * 100) if total_logs > 0 else 0
            
            # Hourly trend - last 24 hours
            cursor = await db.execute(f"""
                SELECT 
                    strftime('%Y-%m-%d %H:00:00', timestamp_utc) as hour,
                    COUNT(*) as count,
                    SUM(CASE WHEN severity >= 4 THEN 1 ELSE 0 END) as errors
                FROM (
                    SELECT timestamp_utc, severity FROM logs_stream 
                    WHERE tenant_id = ? {service_filter} {time_filter}
                    ORDER BY timestamp_utc DESC
                    LIMIT {MAX_SCAN_ROWS}
                )
                GROUP BY hour
                ORDER BY hour DESC
                LIMIT 24
            """, params)
            rows = await cursor.fetchall()
            hourly_trend = [
                {"hour": row["hour"], "count": row["count"], "errors": row["errors"]}
                for row in rows
            ]
            hourly_trend.reverse()
            
            # Template counts - simple query
            cursor = await db.execute("""
                SELECT COUNT(*) as total, 
                       SUM(CASE WHEN embedding_state = 'ready' THEN 1 ELSE 0 END) as embedded
                FROM log_templates WHERE tenant_id = ?
            """, [tenant_id])
            row = await cursor.fetchone()
            total_templates = row["total"] if row else 0
            embedded_templates = row["embedded"] if row else 0
            
            # Services count
            cursor = await db.execute("""
                SELECT COUNT(DISTINCT service_name) as count 
                FROM (SELECT service_name FROM logs_stream WHERE tenant_id = ? LIMIT 10000)
            """, [tenant_id])
            row = await cursor.fetchone()
            services_count = row["count"] if row else 0
            
            template_efficiency = total_logs / total_templates if total_templates > 0 else 0
            
            return {
                "total_logs": total_logs,
                "total_templates": total_templates,
                "embedded_templates": embedded_templates,
                "embedding_coverage": round(embedded_templates / total_templates * 100, 1) if total_templates > 0 else 0,
                "services_count": services_count,
                "severity_breakdown": {
                    "critical": severity_breakdown.get(5, 0),
                    "error": severity_breakdown.get(4, 0),
                    "warning": severity_breakdown.get(3, 0),
                    "info": severity_breakdown.get(2, 0),
                    "debug": severity_breakdown.get(1, 0),
                },
                "error_rate": round(error_rate, 2),
                "warning_rate": round(warn_count / total_logs * 100, 2) if total_logs > 0 else 0,
                "template_efficiency": round(template_efficiency, 1),
                "hourly_trend": hourly_trend,
                "is_estimated": is_estimated,
            }
        except Exception as e:
            logger.error(f"Performance metrics error: {e}")
            return {
                "total_logs": 0, "total_templates": 0, "embedded_templates": 0,
                "embedding_coverage": 0, "services_count": 0,
                "severity_breakdown": {"critical": 0, "error": 0, "warning": 0, "info": 0, "debug": 0},
                "error_rate": 0, "warning_rate": 0, "template_efficiency": 0,
                "hourly_trend": [], "error": str(e),
            }
    
    async def get_security_metrics(
        self,
        tenant_id: Optional[str] = None,
        service_name: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get security metrics by analyzing log content for patterns.
        Uses Python string matching on a sample for speed.
        """
        if tenant_id is None:
            tenant_id = settings.tenant_id_default
            
        db = await get_db()
        
        params: List[Any] = [tenant_id]
        service_filter = ""
        if service_name:
            service_filter = "AND service_name = ?"
            params.append(service_name)
        
        time_filter = ""
        if from_time:
            time_filter += " AND timestamp_utc >= ?"
            params.append(from_time)
        if to_time:
            time_filter += " AND timestamp_utc <= ?"
            params.append(to_time)
        
        # Security pattern keywords (lowercase for matching)
        # These patterns are designed to catch common log messages
        patterns = {
            "authentication_failures": [
                "failed password", "authentication fail", "login fail", "invalid credential",
                "access denied", "unauthorized", "permission denied", "auth fail", "bad password",
                "invalid password", "wrong password", "login denied", "auth error", "authfail",
                "login_error", "invalid user", "invalid login", "authentication error",
                "login failed", "session expired", "invalid token", "token expired"
            ],
            "brute_force_indicators": [
                "too many", "repeated fail", "blocked", "rate limit", "lockout",
                "multiple fail", "max attempt", "account locked", "throttl", "ban",
                "consecutive fail", "attempt limit", "bruteforce", "brute force"
            ],
            "suspicious_access": [
                "injection", "malicious", "exploit", "attack", "suspicious",
                "forbidden", "violation", "intrusion", "hack", "threat", "vulnerability",
                "sql injection", "xss", "csrf", "malware", "phishing", "scanner",
                "directory traversal", "../", "shell", "cmd=", "exec("
            ],
            "network_issues": [
                "connection refused", "timeout", "unreachable", "connection reset",
                "port scan", "network error", "dns fail", "socket error", "connection fail",
                "host not found", "no route", "connection timed out", "connection closed",
                "reset by peer", "broken pipe", "network unreachable", "proxy error"
            ],
            "system_errors": [
                "error state", "error]", "[error", "critical", "fatal", "panic", "crash", 
                "segfault", "out of memory", "oom", "kernel", "core dump", "stack trace", 
                "exception", "null pointer", "500 ", "502 ", "503 ", "504 ",
                "internal server error", "service unavailable", "mod_jk", "worker"
            ]
        }
        
        try:
            # First, get diverse sample by sampling from each service proportionally
            # This ensures we don't just get logs from the most recent (usually access_log)
            all_rows = []
            
            # Get list of services first
            services_cursor = await db.execute("""
                SELECT DISTINCT service_name FROM logs_stream WHERE tenant_id = ?
            """, [tenant_id])
            services = await services_cursor.fetchall()
            services = list(services)
            
            # Sample from each service (roughly 1000 logs each, max 20 services)
            services_to_sample = [s["service_name"] for s in services[:20]]
            samples_per_service = max(500, 10000 // len(services_to_sample)) if services_to_sample else 10000
            
            for svc in services_to_sample:
                svc_params: List[Any] = [tenant_id, svc]
                svc_filter = service_filter
                svc_time_filter = time_filter
                
                # For specific service query, skip the extra service filter
                if service_name:
                    if svc != service_name:
                        continue
                    svc_params = params.copy()
                else:
                    svc_filter = "AND service_name = ?"
                    # Rebuild time params for this query
                    if from_time:
                        svc_time_filter = " AND timestamp_utc >= ?"
                        svc_params.append(from_time)
                    if to_time:
                        svc_time_filter += " AND timestamp_utc <= ?"
                        svc_params.append(to_time)
                
                cursor = await db.execute(f"""
                    SELECT body_raw, timestamp_utc, service_name, severity
                    FROM logs_stream 
                    WHERE tenant_id = ? {svc_filter} {svc_time_filter}
                    ORDER BY RANDOM()
                    LIMIT ?
                """, svc_params + [samples_per_service])
                svc_rows = await cursor.fetchall()
                all_rows.extend(svc_rows)
            
            rows = all_rows
            
            # Analyze logs in Python (fast)
            security_counts = {cat: 0 for cat in patterns.keys()}
            security_examples: Dict[str, List[Dict]] = {cat: [] for cat in patterns.keys()}
            
            for row in rows:
                body = (row["body_raw"] or "").lower()
                
                for category, keywords in patterns.items():
                    for keyword in keywords:
                        if keyword in body:
                            security_counts[category] += 1
                            # Store up to 3 examples per category
                            if len(security_examples[category]) < 3:
                                security_examples[category].append({
                                    "message": row["body_raw"][:200] if row["body_raw"] else "",
                                    "timestamp": row["timestamp_utc"],
                                    "service": row["service_name"],
                                    "severity": row["severity"]
                                })
                            break  # Count each log only once per category
            
            # Calculate totals
            total_security_events = sum(security_counts.values())
            total_logs = len(rows) if rows else 1
            
            # Security score calculation
            issue_ratio = total_security_events / total_logs if total_logs > 0 else 0
            security_score = max(0, min(100, 100 - (issue_ratio * 100)))
            
            # Risk level based on score and specific threats
            if security_counts["suspicious_access"] > 10 or security_counts["brute_force_indicators"] > 20:
                risk_level = "CRITICAL"
            elif security_score >= 90:
                risk_level = "LOW"
            elif security_score >= 70:
                risk_level = "MEDIUM"
            elif security_score >= 50:
                risk_level = "HIGH"
            else:
                risk_level = "CRITICAL"
            
            # Top affected services
            cursor = await db.execute(f"""
                SELECT service_name, COUNT(*) as count
                FROM (
                    SELECT service_name FROM logs_stream 
                    WHERE tenant_id = ? {service_filter} {time_filter}
                    AND severity >= 4
                    LIMIT 10000
                )
                GROUP BY service_name
                ORDER BY count DESC
                LIMIT 10
            """, params)
            affected_rows = await cursor.fetchall()
            top_affected_services = [
                {"service": r["service_name"], "error_count": r["count"]}
                for r in affected_rows
            ]
            
            return {
                "security_score": round(security_score, 1),
                "risk_level": risk_level,
                "total_security_events": total_security_events,
                "categories": security_counts,
                "examples": security_examples,
                "top_affected_services": top_affected_services,
                "recommendations": self._generate_security_recommendations(security_counts, risk_level),
            }
        except Exception as e:
            logger.error(f"Security metrics error: {e}")
            return {
                "security_score": 100, "risk_level": "LOW", "total_security_events": 0,
                "categories": {"authentication_failures": 0, "brute_force_indicators": 0,
                              "suspicious_access": 0, "network_issues": 0, "system_errors": 0},
                "examples": {}, "top_affected_services": [],
                "recommendations": [f"Unable to analyze: {str(e)}"],
            }
    
    def _generate_security_recommendations(self, counts: Dict[str, int], risk_level: str) -> List[str]:
        """Generate detailed security recommendations based on findings."""
        recommendations = []
        
        if counts.get("authentication_failures", 0) > 10:
            recommendations.append("🔐 High authentication failures detected. Review password policies and consider implementing MFA.")
        
        if counts.get("brute_force_indicators", 0) > 5:
            recommendations.append("⚠️ Potential brute force activity. Implement rate limiting and account lockout policies.")
        
        if counts.get("suspicious_access", 0) > 0:
            recommendations.append("🚨 Suspicious access patterns detected. Review firewall rules and access controls immediately.")
        
        if counts.get("network_issues", 0) > 20:
            recommendations.append("🌐 Multiple network issues detected. Check network infrastructure and DNS configuration.")
        
        if counts.get("system_errors", 0) > 50:
            recommendations.append("💻 High system error count. Review application stability and resource allocation.")
        
        if risk_level == "CRITICAL":
            recommendations.insert(0, "🚨 CRITICAL: Immediate security review required!")
        elif risk_level == "HIGH":
            recommendations.insert(0, "⚠️ HIGH RISK: Review security patterns soon.")
        
        if not recommendations:
            recommendations.append("✅ No significant security concerns detected in analyzed logs.")
        
        return recommendations
    
    def _generate_recommendations(self, error_count: int, risk_level: str) -> List[str]:
        """Generate recommendations based on error count."""
        recommendations = []
        
        if error_count > 1000:
            recommendations.append("High error volume detected. Review application logs for patterns.")
        elif error_count > 100:
            recommendations.append("Moderate error count. Monitor for trends.")
        
        if risk_level == "CRITICAL":
            recommendations.insert(0, "⚠️ CRITICAL: Immediate review recommended!")
        elif risk_level == "HIGH":
            recommendations.insert(0, "⚠️ HIGH: Review error patterns soon.")
        
        if not recommendations:
            recommendations.append("✅ System health looks good.")
        
        return recommendations
    
    async def get_service_health(
        self,
        tenant_id: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get health status for each service with proper aggregation."""
        if tenant_id is None:
            tenant_id = settings.tenant_id_default
            
        db = await get_db()
        
        params: List[Any] = [tenant_id]
        time_filter = ""
        if from_time:
            time_filter += " AND timestamp_utc >= ?"
            params.append(from_time)
        if to_time:
            time_filter += " AND timestamp_utc <= ?"
            params.append(to_time)
        
        try:
            # Get ALL services with their stats (not limited by sample)
            cursor = await db.execute(f"""
                SELECT 
                    service_name,
                    COUNT(*) as total_logs,
                    SUM(CASE WHEN severity >= 4 THEN 1 ELSE 0 END) as errors,
                    SUM(CASE WHEN severity = 3 THEN 1 ELSE 0 END) as warnings,
                    MAX(timestamp_utc) as last_seen
                FROM logs_stream 
                WHERE tenant_id = ? {time_filter}
                GROUP BY service_name
                ORDER BY total_logs DESC
                LIMIT 100
            """, params)
            
            rows = await cursor.fetchall()
            services = []
            
            for row in rows:
                total = row["total_logs"] or 0
                errors = row["errors"] or 0
                warnings = row["warnings"] or 0
                error_rate = (errors / total * 100) if total > 0 else 0
                
                if error_rate < 1:
                    status = "healthy"
                elif error_rate < 5:
                    status = "warning"
                elif error_rate < 10:
                    status = "degraded"
                else:
                    status = "critical"
                
                services.append({
                    "service_name": row["service_name"],
                    "status": status,
                    "total_logs": total,
                    "errors": errors,
                    "warnings": warnings,
                    "error_rate": round(error_rate, 2),
                    "last_seen": row["last_seen"],
                })
            
            return services
        except Exception as e:
            logger.error(f"Service health error: {e}")
            return []


# Global service instance
_metrics_service: Optional[MetricsService] = None


def get_metrics_service() -> MetricsService:
    """Get global metrics service instance."""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service
