"""Tests for chat API contract."""

import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest, ChatResponse, Citation


class TestChatRequest:
    """Test ChatRequest schema."""

    def test_valid_request(self):
        """Test valid chat request."""
        request = ChatRequest(
            service_name="my-service",
            question="What errors occurred?",
            **{"from": "2024-01-15T00:00:00Z", "to": "2024-01-15T23:59:59Z"}
        )
        
        assert request.service_name == "my-service"
        assert request.question == "What errors occurred?"

    def test_request_without_time_range(self):
        """Test chat request without time range (should fail)."""
        # from and to are required - using from_time/to_time aliases
        with pytest.raises(ValidationError):
            ChatRequest(
                service_name="my-service",
                question="What happened?",
                from_time=None,  # type: ignore
                to_time=None     # type: ignore
            )

    def test_missing_service_name(self):
        """Test that service_name is required."""
        with pytest.raises(ValidationError):
            ChatRequest(
                question="What happened?",
                **{"from": "2024-01-15T00:00:00Z", "to": "2024-01-15T23:59:59Z"}
            )

    def test_missing_question(self):
        """Test that question is required."""
        with pytest.raises(ValidationError):
            ChatRequest(
                service_name="my-service",
                **{"from": "2024-01-15T00:00:00Z", "to": "2024-01-15T23:59:59Z"}
            )

    def test_empty_question(self):
        """Test that empty question raises error."""
        with pytest.raises(ValidationError):
            ChatRequest(
                service_name="my-service", 
                question="",
                **{"from": "2024-01-15T00:00:00Z", "to": "2024-01-15T23:59:59Z"}
            )


class TestChatResponse:
    """Test ChatResponse schema."""

    def test_valid_response(self):
        """Test valid chat response."""
        response = ChatResponse(
            answer="There were 5 connection errors.",
            citations=[
                Citation(
                    type="template",
                    template_hash=12345,
                    template_text="Connection error <*>"
                )
            ],
            confidence="high",
            next_steps=["Check network connectivity"]
        )
        
        assert response.answer == "There were 5 connection errors."
        assert len(response.citations) == 1
        assert response.confidence == "high"

    def test_response_with_log_citation(self):
        """Test response with log citation."""
        response = ChatResponse(
            answer="Error found in logs.",
            citations=[
                Citation(type="log", log_id=42)
            ],
            confidence="medium"
        )
        
        assert response.citations[0].type == "log"
        assert response.citations[0].log_id == 42

    def test_response_defaults(self):
        """Test response default values."""
        response = ChatResponse(answer="Test answer")
        
        assert response.citations == []
        assert response.confidence == "medium"
        assert response.next_steps == []
        assert response.metadata == {}


class TestCitation:
    """Test Citation schema."""

    def test_template_citation(self):
        """Test template citation."""
        citation = Citation(
            type="template",
            service_name="my-service",
            template_hash=12345,
            template_text="User <*> logged in from <*>",
            relevance="Contains error patterns"
        )
        
        assert citation.type == "template"
        assert citation.template_hash == 12345

    def test_log_citation(self):
        """Test log citation."""
        citation = Citation(
            type="log",
            log_id=42
        )
        
        assert citation.type == "log"
        assert citation.log_id == 42

    def test_citation_optional_fields(self):
        """Test citation with only required fields."""
        citation = Citation(type="template")
        
        assert citation.type == "template"
        assert citation.template_hash is None
        assert citation.log_id is None
