"""Tests for security metrics service."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

# Mock the database module before importing the service
import sys
sys.modules['app.storage.db'] = MagicMock()
sys.modules['app.core.config'] = MagicMock()
sys.modules['app.core.logging'] = MagicMock()

from app.services.security_metrics_service import (
    SecurityMetricsService,
    _load_patterns,
)


class TestSecurityPatternsLoading:
    """Test loading security patterns from JSON."""
    
    def test_patterns_file_exists(self):
        """Verify security_patterns.json exists."""
        patterns_path = Path(__file__).parent.parent / "app" / "services" / "security_patterns.json"
        assert patterns_path.exists(), f"Patterns file not found at {patterns_path}"
    
    def test_patterns_valid_json(self):
        """Verify patterns file is valid JSON."""
        patterns_path = Path(__file__).parent.parent / "app" / "services" / "security_patterns.json"
        with open(patterns_path, 'r') as f:
            data = json.load(f)
        
        # Verify required keys
        assert "patterns" in data
        assert "scoring" in data
        assert "risk_levels" in data
    
    def test_patterns_have_required_fields(self):
        """Verify each pattern category has required fields."""
        patterns_path = Path(__file__).parent.parent / "app" / "services" / "security_patterns.json"
        with open(patterns_path, 'r') as f:
            data = json.load(f)
        
        for category, config in data["patterns"].items():
            assert "keywords" in config, f"{category} missing keywords"
            assert "description" in config, f"{category} missing description"
            assert "severity_boost" in config, f"{category} missing severity_boost"
            assert isinstance(config["keywords"], list), f"{category} keywords should be a list"
            assert len(config["keywords"]) > 0, f"{category} has no keywords"
    
    def test_risk_levels_complete(self):
        """Verify all risk levels are defined."""
        patterns_path = Path(__file__).parent.parent / "app" / "services" / "security_patterns.json"
        with open(patterns_path, 'r') as f:
            data = json.load(f)
        
        expected_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        for level in expected_levels:
            assert level in data["risk_levels"], f"Missing risk level: {level}"


class TestSecurityMetricsService:
    """Test SecurityMetricsService class."""
    
    def test_service_initialization(self):
        """Test service loads patterns correctly."""
        service = SecurityMetricsService()
        
        assert service.patterns is not None
        assert len(service.patterns) > 0
        assert "authentication_failures" in service.patterns
        assert "brute_force_indicators" in service.patterns
        assert "suspicious_access" in service.patterns
    
    def test_calculate_score_base(self):
        """Test score calculation with no issues."""
        service = SecurityMetricsService()
        
        counts = {cat: 0 for cat in service.patterns.keys()}
        score = service._calculate_score(counts, 1000)
        
        assert score == 100.0, "Score should be 100 with no issues"
    
    def test_calculate_score_with_issues(self):
        """Test score calculation with security issues."""
        service = SecurityMetricsService()
        
        counts = {cat: 0 for cat in service.patterns.keys()}
        counts["authentication_failures"] = 50
        counts["suspicious_access"] = 10
        
        score = service._calculate_score(counts, 1000)
        
        assert score < 100.0, "Score should decrease with issues"
        assert score >= 0.0, "Score should not be negative"
    
    def test_determine_risk_level_low(self):
        """Test risk level determination for low risk."""
        service = SecurityMetricsService()
        
        counts = {cat: 0 for cat in service.patterns.keys()}
        risk = service._determine_risk_level(95, counts)
        
        assert risk == "LOW"
    
    def test_determine_risk_level_critical_threshold(self):
        """Test critical risk level for high suspicious access."""
        service = SecurityMetricsService()
        
        counts = {cat: 0 for cat in service.patterns.keys()}
        counts["suspicious_access"] = 100  # Above critical threshold
        
        risk = service._determine_risk_level(50, counts)
        
        assert risk == "CRITICAL"
    
    def test_generate_recommendations_empty(self):
        """Test recommendations with no issues."""
        service = SecurityMetricsService()
        
        counts = {cat: 0 for cat in service.patterns.keys()}
        recommendations = service._generate_recommendations(counts, "LOW")
        
        assert len(recommendations) >= 1
        assert any("No significant" in r or "✅" in r for r in recommendations)
    
    def test_generate_recommendations_auth_failures(self):
        """Test recommendations for authentication failures."""
        service = SecurityMetricsService()
        
        counts = {cat: 0 for cat in service.patterns.keys()}
        counts["authentication_failures"] = 50
        
        recommendations = service._generate_recommendations(counts, "HIGH")
        
        assert any("authentication" in r.lower() or "mfa" in r.lower() for r in recommendations)
    
    def test_generate_recommendations_brute_force(self):
        """Test recommendations for brute force indicators."""
        service = SecurityMetricsService()
        
        counts = {cat: 0 for cat in service.patterns.keys()}
        counts["brute_force_indicators"] = 20
        
        recommendations = service._generate_recommendations(counts, "HIGH")
        
        assert any("brute" in r.lower() or "lockout" in r.lower() for r in recommendations)
    
    def test_analyze_rows_pattern_matching(self):
        """Test pattern matching in log analysis."""
        service = SecurityMetricsService()
        
        # Create mock rows
        mock_rows = [
            {
                "body_raw": "Failed password for invalid user admin from 192.168.1.1",
                "timestamp_utc": "2024-01-01T00:00:00Z",
                "service_name": "sshd",
                "severity": 4
            },
            {
                "body_raw": "Connection refused from 10.0.0.1",
                "timestamp_utc": "2024-01-01T00:00:01Z",
                "service_name": "nginx",
                "severity": 3
            },
            {
                "body_raw": "Normal operation log message",
                "timestamp_utc": "2024-01-01T00:00:02Z",
                "service_name": "app",
                "severity": 2
            },
        ]
        
        results = service._analyze_rows(mock_rows)
        
        assert results["total_logs_analyzed"] == 3
        assert results["categories"]["authentication_failures"] > 0
        assert results["categories"]["network_issues"] > 0
        assert "security_score" in results
        assert "risk_level" in results
    
    def test_empty_result_structure(self):
        """Test empty result structure on error."""
        service = SecurityMetricsService()
        
        result = service._empty_result("Test error")
        
        assert result["security_score"] == 100
        assert result["risk_level"] == "LOW"
        assert result["total_security_events"] == 0
        assert "error" in result
        assert result["error"] == "Test error"


class TestPatternKeywords:
    """Test specific pattern keyword detection."""
    
    def test_auth_failure_keywords(self):
        """Test authentication failure keyword detection."""
        patterns_path = Path(__file__).parent.parent / "app" / "services" / "security_patterns.json"
        with open(patterns_path, 'r') as f:
            data = json.load(f)
        
        auth_keywords = data["patterns"]["authentication_failures"]["keywords"]
        
        test_messages = [
            ("Failed password for root", True),
            ("authentication failed for user admin", True),
            ("login failed from 192.168.1.1", True),
            ("access denied to /admin", True),
            ("User logged in successfully", False),
        ]
        
        for message, should_match in test_messages:
            message_lower = message.lower()
            matched = any(kw in message_lower for kw in auth_keywords)
            assert matched == should_match, f"'{message}' matching failed"
    
    def test_suspicious_access_keywords(self):
        """Test suspicious access keyword detection."""
        patterns_path = Path(__file__).parent.parent / "app" / "services" / "security_patterns.json"
        with open(patterns_path, 'r') as f:
            data = json.load(f)
        
        sus_keywords = data["patterns"]["suspicious_access"]["keywords"]
        
        test_messages = [
            ("SQL injection attempt detected", True),
            ("XSS attack blocked", True),
            ("Possible directory traversal ../../../etc/passwd", True),
            ("Normal page request /index.html", False),
        ]
        
        for message, should_match in test_messages:
            message_lower = message.lower()
            matched = any(kw in message_lower for kw in sus_keywords)
            assert matched == should_match, f"'{message}' matching failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
