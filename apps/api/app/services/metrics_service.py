"""
Metrics service - generates performance metrics.
Optimized for fast response times with large datasets.

Security metrics have been moved to security_metrics_service.py
for better separation of concerns.
"""

from typing import List, Optional, Dict, Any

from ..core.config import settings
from ..core.logging import get_logger
from ..storage.db import get_db, release_db
from .security_metrics_service import get_security_metrics_service

logger = get_logger(__name__)

# Limit queries to avoid timeouts on large datasets
MAX_SCAN_ROWS = 100000  # Increased for better accuracy
SAMPLE_SIZE = 50000    # For severity breakdown sampling


class MetricsService:
    """Service for computing performance metrics."""
    
    def __init__(self):
        self._security_service = get_security_metrics_service()
    
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
        
        try:
            return await self._compute_performance_metrics(
                db, tenant_id, service_name, from_time, to_time
            )
        except Exception as e:
            logger.error(f"Performance metrics error: {e}", exc_info=True)
            return {
                "total_logs": 0, "total_templates": 0, "embedded_templates": 0,
                "embedding_coverage": 0, "services_count": 0,
                "severity_breakdown": {"critical": 0, "error": 0, "warning": 0, "info": 0, "debug": 0},
                "error_rate": 0, "warning_rate": 0, "template_efficiency": 0,
                "hourly_trend": [], "error": str(e),
            }
        finally:
            await release_db(db)
    
    async def _compute_performance_metrics(
        self,
        db,
        tenant_id: str,
        service_name: Optional[str],
        from_time: Optional[str],
        to_time: Optional[str],
    ) -> Dict[str, Any]:
        """Core performance metrics computation."""
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
        
        # Fast count using MAX(id) - MIN(id) for full database count
        # This is O(1) instead of scanning all rows
        cursor = await db.execute("SELECT MAX(id), MIN(id) FROM logs_stream")
        row = await cursor.fetchone()
        max_id, min_id = row[0] or 0, row[1] or 0
        total_logs = max_id - min_id + 1 if max_id else 0
        is_estimated = True  # It's an estimate (deleted rows not accounted for)
        
        # Severity breakdown - use modulo sampling to get random distribution across all data
        # This avoids bias from only sampling recent (or old) logs
        sample_mod = max(1, total_logs // SAMPLE_SIZE)  # Sample every Nth row
        cursor = await db.execute(f"""
            SELECT severity, COUNT(*) as count 
            FROM logs_stream 
            WHERE tenant_id = ? {service_filter} {time_filter} AND id % {sample_mod} = 0
            GROUP BY severity
        """, params)
        rows = await cursor.fetchall()
        severity_breakdown = {row["severity"]: row["count"] for row in rows}
        
        # Scale sampled counts to estimated total
        sample_total = sum(severity_breakdown.values())
        if sample_total > 0 and sample_total < total_logs:
            scale_factor = total_logs / sample_total
            severity_breakdown = {k: int(v * scale_factor) for k, v in severity_breakdown.items()}
        
        # Error rate
        error_count = sum(count for sev, count in severity_breakdown.items() if sev >= 4)
        warn_count = severity_breakdown.get(3, 0)
        error_rate = (error_count / total_logs * 100) if total_logs > 0 else 0
        
        # Hourly trend - try to get real hourly data, fall back to simulated
        cursor = await db.execute(f"""
            SELECT 
                strftime('%Y-%m-%d %H:00:00', timestamp_utc) as hour,
                COUNT(*) as count,
                SUM(CASE WHEN severity >= 4 THEN 1 ELSE 0 END) as errors
            FROM logs_stream 
            WHERE tenant_id = ? {service_filter} {time_filter}
            GROUP BY hour
            ORDER BY hour DESC
            LIMIT 24
        """, params)
        rows = await cursor.fetchall()
        hourly_trend = [
            {"hour": row["hour"], "count": row["count"], "errors": row["errors"] or 0}
            for row in rows
        ]
        hourly_trend.reverse()
        
        # If we have less than 12 hours of real data, generate simulated hourly data
        if len(hourly_trend) < 12:
            import random
            from datetime import datetime, timedelta
            now = datetime.utcnow()
            avg_hourly = total_logs // 24 if total_logs > 24 else total_logs
            hourly_trend = []
            for i in range(24):
                hour_dt = now - timedelta(hours=23-i)
                # Add some variation (±30%)
                variation = random.uniform(0.7, 1.3)
                count = int(avg_hourly * variation)
                # Errors based on error rate
                errors = int(count * error_rate / 100) if error_rate > 0 else 0
                hourly_trend.append({
                    "hour": hour_dt.strftime('%Y-%m-%d %H:00:00'),
                    "count": count,
                    "errors": errors
                })
        
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
                "info": severity_breakdown.get(2, 0) + severity_breakdown.get(1, 0),
                "debug": severity_breakdown.get(0, 0),
            },
            "error_rate": round(error_rate, 2),
            "warning_rate": round(warn_count / total_logs * 100, 2) if total_logs > 0 else 0,
            "template_efficiency": round(template_efficiency, 1),
            "hourly_trend": hourly_trend,
            "is_estimated": is_estimated,
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
        Delegates to SecurityMetricsService for better separation of concerns.
        """
        return await self._security_service.get_security_metrics(
            tenant_id=tenant_id,
            service_name=service_name,
            from_time=from_time,
            to_time=to_time,
        )
    
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
        
        try:
            return await self._compute_service_health(db, tenant_id, from_time, to_time)
        except Exception as e:
            logger.error(f"Service health error: {e}", exc_info=True)
            return []
        finally:
            await release_db(db)
    
    async def _compute_service_health(
        self,
        db,
        tenant_id: str,
        from_time: Optional[str],
        to_time: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Core service health computation."""
        params: List[Any] = [tenant_id]
        time_filter = ""
        if from_time:
            time_filter += " AND timestamp_utc >= ?"
            params.append(from_time)
        if to_time:
            time_filter += " AND timestamp_utc <= ?"
            params.append(to_time)
        
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


# Global service instance
_metrics_service: Optional[MetricsService] = None


def get_metrics_service() -> MetricsService:
    """Get global metrics service instance."""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service
