"""Tests for input validation schemas."""

import pytest
from pydantic import ValidationError


class TestChatRequestValidation:
    """Test ChatRequest schema validation."""
    
    def test_valid_request(self):
        """Test valid chat request passes validation."""
        from app.schemas.chat import ChatRequest
        
        req = ChatRequest(
            service_name="nginx",
            question="What are the most common error patterns?",
            **{"from": "2024-01-01T00:00:00Z", "to": "2024-01-02T00:00:00Z"}
        )
        
        assert req.service_name == "nginx"
        assert req.question == "What are the most common error patterns?"
    
    def test_empty_question_rejected(self):
        """Test empty question is rejected."""
        from app.schemas.chat import ChatRequest
        
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(
                service_name="nginx",
                question="",
                **{"from": "2024-01-01T00:00:00Z", "to": "2024-01-02T00:00:00Z"}
            )
        
        errors = exc_info.value.errors()
        assert len(errors) > 0
    
    def test_question_too_long_rejected(self):
        """Test overly long question is rejected."""
        from app.schemas.chat import ChatRequest
        
        long_question = "x" * 5001  # Over 5000 char limit
        
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(
                service_name="nginx",
                question=long_question,
                **{"from": "2024-01-01T00:00:00Z", "to": "2024-01-02T00:00:00Z"}
            )
        
        errors = exc_info.value.errors()
        assert any("max_length" in str(e) or "5000" in str(e) for e in errors)
    
    def test_service_name_sanitization(self):
        """Test service name is sanitized."""
        from app.schemas.chat import ChatRequest
        
        req = ChatRequest(
            service_name="nginx-production_v2.0",  # Valid characters
            question="What errors?",
            **{"from": "2024-01-01T00:00:00Z", "to": "2024-01-02T00:00:00Z"}
        )
        
        assert req.service_name == "nginx-production_v2.0"
    
    def test_service_name_invalid_chars_removed(self):
        """Test invalid characters are removed from service name."""
        from app.schemas.chat import ChatRequest
        
        req = ChatRequest(
            service_name="nginx@production#test",  # Invalid @ and # chars
            question="What errors?",
            **{"from": "2024-01-01T00:00:00Z", "to": "2024-01-02T00:00:00Z"}
        )
        
        # @ and # should be stripped
        assert "@" not in req.service_name
        assert "#" not in req.service_name
    
    def test_question_sanitization_whitespace(self):
        """Test question whitespace is normalized."""
        from app.schemas.chat import ChatRequest
        
        req = ChatRequest(
            service_name="nginx",
            question="What   are   the   errors?",  # Multiple spaces
            **{"from": "2024-01-01T00:00:00Z", "to": "2024-01-02T00:00:00Z"}
        )
        
        # Multiple spaces should be collapsed
        assert "   " not in req.question


class TestChatResponseValidation:
    """Test ChatResponse schema."""
    
    def test_valid_response(self):
        """Test valid response structure."""
        from app.schemas.chat import ChatResponse
        
        response = ChatResponse(
            answer="Here's what I found...",
            citations=[],
            confidence="high"
        )
        
        assert response.answer == "Here's what I found..."
        assert response.confidence == "high"
    
    def test_response_with_citations(self):
        """Test response with citations."""
        from app.schemas.chat import ChatResponse, Citation
        
        citation = Citation(
            type="template",
            template_text="Error: connection refused",
            template_hash=12345
        )
        
        response = ChatResponse(
            answer="Connection errors found",
            citations=[citation],
            confidence="medium"
        )
        
        assert len(response.citations) == 1
        assert response.citations[0].type == "template"


class TestSemanticSearchValidation:
    """Test semantic search parameter validation."""
    
    def test_valid_search_params(self):
        """Test valid search parameters."""
        from app.schemas.chat import SemanticSearchParams
        
        params = SemanticSearchParams(
            service_name="nginx",
            q="connection error",
            limit=10,
            **{"from": "2024-01-01T00:00:00Z", "to": "2024-01-02T00:00:00Z"}
        )
        
        assert params.service_name == "nginx"
        assert params.q == "connection error"
        assert params.limit == 10
    
    def test_limit_max_enforced(self):
        """Test limit maximum is enforced."""
        from app.schemas.chat import SemanticSearchParams
        
        with pytest.raises(ValidationError):
            SemanticSearchParams(
                service_name="nginx",
                q="error",
                limit=100,  # Over max of 50
                **{"from": "2024-01-01T00:00:00Z", "to": "2024-01-02T00:00:00Z"}
            )
    
    def test_empty_query_rejected(self):
        """Test empty query is rejected."""
        from app.schemas.chat import SemanticSearchParams
        
        with pytest.raises(ValidationError):
            SemanticSearchParams(
                service_name="nginx",
                q="",  # Empty
                **{"from": "2024-01-01T00:00:00Z", "to": "2024-01-02T00:00:00Z"}
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
