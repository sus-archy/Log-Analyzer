"""
Chat service - handles grounded chat with logs using cross-domain few-shot learning.
"""

from typing import List, Optional

from ..core.config import settings
from ..core.logging import get_logger
from ..llm.ollama_client import get_ollama_client, OllamaError
from ..schemas.chat import ChatRequest, ChatResponse, Citation
from ..schemas.templates import TemplateWithCount
from ..storage.db import get_db
from ..storage.templates_repo import TemplatesRepo
from ..storage.logs_repo import LogsRepo
from .semantic_service import get_semantic_service
from .few_shot_service import get_few_shot_service

logger = get_logger(__name__)

# Enhanced system prompt with cross-domain learning
SYSTEM_PROMPT = """You are an expert log analysis assistant with cross-domain knowledge.

You have been trained on patterns from multiple log domains:
- Web servers (Apache, Nginx)
- Authentication systems (SSH, Kerberos)
- Operating systems (Linux, Windows)
- Networks (connections, firewalls)
- Databases (MySQL, PostgreSQL)
- Distributed systems (Hadoop, Kubernetes)
- Security events (attacks, intrusions)

CROSS-DOMAIN ANALYSIS PRINCIPLES:
1. Patterns in one domain often indicate issues in related domains
2. Authentication failures may correlate with web attacks
3. Network issues cascade to application and database errors
4. System resource issues cause application failures
5. Security events leave traces across multiple log types

RESPONSE FORMAT:
- **Summary**: Brief answer (2-3 sentences)
- **Analysis**: Detailed findings with cross-domain insights
- **Evidence**: Cite logs using [template:HASH] or [log:ID]
- **Cross-Domain Patterns**: Related issues in other domains
- **Recommendations**: Prioritized action items

IMPORTANT: Apply knowledge from example analyses to new situations."""


class ChatService:
    """Service for grounded chat with logs."""
    
    async def chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Generate a grounded response to a log question.
        
        Args:
            request: Chat request with scope and question
            
        Returns:
            ChatResponse with answer and citations
        """
        tenant_id = request.tenant_id or settings.tenant_id_default
        logger.info(f"Chat request: service={request.service_name}, question={request.question[:50]}...")
        
        # Gather evidence
        logger.info("Gathering evidence...")
        evidence_context, citations, log_samples = await self._gather_evidence(
            tenant_id=tenant_id,
            service_name=request.service_name,
            from_time=request.from_time,
            to_time=request.to_time,
            question=request.question,
        )
        logger.info(f"Evidence gathered: {len(citations)} citations, {len(evidence_context)} chars")
        
        # Build prompt with few-shot learning
        user_message = self._build_user_message(
            question=request.question,
            service_name=request.service_name,
            from_time=request.from_time,
            to_time=request.to_time,
            evidence=evidence_context,
            log_samples=log_samples,
        )
        logger.info(f"Few-shot prompt built: {len(user_message)} chars. Calling LLM...")
        
        # Call LLM
        ollama = get_ollama_client()
        
        try:
            answer = await ollama.chat(
                messages=[{"role": "user", "content": user_message}],
                system=SYSTEM_PROMPT,
                temperature=0.3,
            )
        except OllamaError as e:
            logger.error(f"Chat failed: {e}")
            return ChatResponse(
                answer=f"Failed to generate response: {e}",
                citations=citations,
                confidence="low",
                next_steps=["Check if Ollama is running", "Try again later"],
            )
        
        # Parse confidence and next steps from response
        confidence = self._extract_confidence(answer)
        next_steps = self._extract_next_steps(answer)
        
        return ChatResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            next_steps=next_steps,
            metadata={
                "service_name": request.service_name,
                "time_range": f"{request.from_time} to {request.to_time}",
            },
        )
    
    async def _gather_evidence(
        self,
        tenant_id: str,
        service_name: str,
        from_time: str,
        to_time: str,
        question: str,
    ) -> tuple[str, List[Citation]]:
        """
        Gather evidence from templates and logs.
        
        Returns:
            Tuple of (evidence text, citations list)
        """
        db = await get_db()
        templates_repo = TemplatesRepo(db)
        logs_repo = LogsRepo(db)
        
        citations: List[Citation] = []
        evidence_parts: List[str] = []
        
        logger.info(f"Gathering evidence: tenant={tenant_id}, service={service_name}, from={from_time}, to={to_time}")
        
        # 1. Get top error templates (reduced for speed)
        try:
            top_templates = await templates_repo.get_top_templates(
                tenant_id=tenant_id,
                service_name=service_name,
                from_time=from_time,
                to_time=to_time,
                severity_min=0,  # All severities
                limit=5,  # Reduced for speed
            )
            logger.info(f"Got {len(top_templates)} top templates")
        except Exception as e:
            logger.warning(f"Failed to get top templates: {e}", exc_info=True)
            top_templates = []
        
        if top_templates:
            evidence_parts.append("=== TOP LOG PATTERNS ===")
            for t in top_templates:
                evidence_parts.append(
                    f"[template:{t.template_hash}] (count={t.count}): {t.template_text}"
                )
                citations.append(Citation(
                    type="template",
                    service_name=service_name,
                    template_hash=t.template_hash,
                    template_text=t.template_text,
                    relevance="top pattern",
                ))
        
        # 2. Try semantic search only if we have embeddings
        try:
            semantic_service = get_semantic_service()
            semantic_results = await semantic_service.search(
                query=question,
                tenant_id=tenant_id,
                service_name=service_name,
                from_time=from_time,
                to_time=to_time,
                limit=3,  # Reduced for speed
            )
            
            if semantic_results.results:
                evidence_parts.append("\n=== RELATED PATTERNS ===")
                for r in semantic_results.results:
                    if any(c.template_hash == r.template_hash for c in citations):
                        continue
                    evidence_parts.append(
                        f"[template:{r.template_hash}] (relevance={r.score:.2f}): {r.template_text}"
                    )
                    citations.append(Citation(
                        type="template",
                        service_name=service_name,
                        template_hash=r.template_hash,
                        template_text=r.template_text,
                        relevance="semantic match",
                    ))
        except Exception as e:
            logger.warning(f"Semantic search failed (may not have embeddings): {e}")
        
        # 3. Get a few sample logs
        evidence_parts.append("\n=== SAMPLE LOGS ===")
        sample_count = 0
        max_samples = 5  # Reduced for speed
        
        for template in top_templates[:3]:
            if sample_count >= max_samples:
                break
            
            try:
                sample_logs = await logs_repo.get_sample_logs_for_template(
                    tenant_id=tenant_id,
                    service_name=service_name,
                    template_hash=template.template_hash,
                    from_time=from_time,
                    to_time=to_time,
                    limit=2,
                )
                
                for log in sample_logs:
                    if sample_count >= max_samples:
                        break
                    evidence_parts.append(
                        f"[log:{log.id}] {log.timestamp_utc} [{log.severity_name}]: {log.body_raw[:150]}"
                    )
                    citations.append(Citation(
                        type="log",
                        log_id=log.id,
                        relevance=f"example",
                    ))
                    sample_count += 1
            except Exception as e:
                logger.warning(f"Failed to get sample logs: {e}")
        
        evidence_text = "\\n".join(evidence_parts) if evidence_parts else "No log data found in the specified scope."
        
        # Collect sample log texts for domain detection
        sample_texts = [p for p in evidence_parts if "[template:" in p or "[log:" in p]
        
        return evidence_text, citations, sample_texts
    
    def _build_user_message(
        self,
        question: str,
        service_name: str,
        from_time: str,
        to_time: str,
        evidence: str,
        log_samples: List[str] = None,
    ) -> str:
        """Build the user message with few-shot examples and cross-domain context."""
        # Use few-shot service to build enhanced prompt
        few_shot_service = get_few_shot_service()
        
        # Build few-shot prompt with cross-domain knowledge
        enhanced_prompt = few_shot_service.build_few_shot_prompt(
            question=question,
            service_name=service_name,
            log_samples=log_samples,
            evidence_context=evidence,
            max_examples=2,  # Include 2 most relevant examples
        )
        
        # Add time context
        time_context = f"\nTime Range: {from_time} to {to_time}"
        
        return enhanced_prompt + time_context
    
    def _extract_confidence(self, answer: str) -> str:
        """Extract confidence level from answer."""
        answer_lower = answer.lower()
        
        if "confidence: high" in answer_lower or "high confidence" in answer_lower:
            return "high"
        elif "confidence: low" in answer_lower or "low confidence" in answer_lower:
            return "low"
        else:
            return "medium"
    
    def _extract_next_steps(self, answer: str) -> List[str]:
        """Extract next steps from answer."""
        steps = []
        
        # Look for numbered steps
        lines = answer.split("\n")
        in_next_steps = False
        
        for line in lines:
            line = line.strip()
            
            if "next steps" in line.lower():
                in_next_steps = True
                continue
            
            if in_next_steps:
                # Stop at next section
                if line.startswith("===") or (line and line.isupper() and ":" in line):
                    break
                
                # Extract numbered or bulleted items
                if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                    # Clean up the line
                    step = line.lstrip("0123456789.-*) ").strip()
                    if step:
                        steps.append(step)
        
        return steps


# Global service instance
_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get global chat service instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
