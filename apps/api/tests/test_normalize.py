"""Tests for log normalization logic."""

import pytest
from app.parsers.text_parser import parse_text_line


class TestNormalizeLogEvent:
    """Test parse_text_line function (normalize_log_event)."""

    def test_normalize_syslog_format(self):
        """Test normalizing syslog format logs."""
        raw = "Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure"
        result = parse_text_line(raw, default_service="sshd")
        
        assert result is not None
        assert result["body_raw"] == raw
        assert result["service_name"] == "sshd"
        assert result["severity"] >= 0

    def test_normalize_with_severity_extraction(self):
        """Test severity extraction from log content."""
        test_cases = [
            ("DEBUG: Starting process", 1),
            ("INFO: Process started", 2),
            ("WARNING: Low memory", 3),
            ("ERROR: Connection failed", 4),
            ("FATAL: System crash", 5),
        ]
        
        for raw, expected_severity in test_cases:
            result = parse_text_line(raw)
            assert result is not None
            assert result["severity"] == expected_severity, f"Failed for: {raw}"

    def test_normalize_with_custom_service(self):
        """Test service name override."""
        raw = "Some log message"
        result = parse_text_line(raw, default_service="my-service")
        
        assert result is not None
        assert result["service_name"] == "my-service"

    def test_normalize_default_severity(self):
        """Test default severity for logs without severity indicators."""
        raw = "Just a plain message without severity"
        result = parse_text_line(raw)
        
        assert result is not None
        assert result["severity"] == 2  # Default INFO

    def test_normalize_preserves_raw_message(self):
        """Test that raw message is preserved."""
        raw = "Original message with special chars: @#$%^&*()"
        result = parse_text_line(raw)
        
        assert result is not None
        assert result["body_raw"] == raw

    def test_normalize_case_insensitive_severity(self):
        """Test case-insensitive severity extraction."""
        test_cases = [
            ("error: something failed", 4),
            ("ERROR: something failed", 4),
            ("Error: something failed", 4),
        ]
        
        for raw, expected_severity in test_cases:
            result = parse_text_line(raw)
            assert result is not None
            assert result["severity"] == expected_severity

    def test_normalize_returns_dict(self):
        """Test that result is always a dict."""
        raw = "Test message"
        result = parse_text_line(raw)
        
        assert isinstance(result, dict)
        assert "timestamp_utc" in result
        assert "severity" in result
        assert "message" in result
        assert "service_name" in result
        assert "body_raw" in result

    def test_normalize_timestamp_field(self):
        """Test timestamp field is present."""
        raw = "2024-01-15 10:30:00 INFO Starting"
        result = parse_text_line(raw)
        
        assert result["timestamp_utc"] is not None
