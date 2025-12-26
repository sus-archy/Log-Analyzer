"""
Chat routes for grounded chat with logs using cross-domain few-shot learning.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from ..core.logging import get_logger
from ..core.cache import get_cache, CacheTTL
from ..schemas.chat import ChatRequest, ChatResponse
from ..services.chat_service import get_chat_service
from ..services.embedding_service import get_embedding_service
from ..services.few_shot_service import get_few_shot_service

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with logs for diagnosis using cross-domain few-shot learning.
    
    The AI uses patterns learned from multiple domains (web, auth, network, etc.)
    to provide better analysis even for unfamiliar log types.
    
    Requires a scope (service_name + time range) and a question.
    Returns a grounded answer with citations to templates and logs.
    """
    service = get_chat_service()
    
    try:
        return await service.chat(request)
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed")
async def process_embeddings(max_templates: int = 100):
    """
    Process pending template embeddings.
    
    Called manually or by a background job to generate embeddings
    for templates that don't have them yet.
    """
    service = get_embedding_service()
    
    try:
        count = await service.process_pending_templates(max_templates=max_templates)
        return {"processed": count}
    except Exception as e:
        logger.error(f"Embedding processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@router.get("/embed/stats")
async def get_embedding_stats():
    """
    Get embedding generation statistics.
    
    Returns total templates, embedded count, and percentage complete.
    """
    cache = get_cache()
    cache_key = "embed_stats"
    
    # Try cache first for fast response
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached
    
    service = get_embedding_service()
    
    try:
        stats = await service.get_embedding_stats()
        await cache.set(cache_key, stats, CacheTTL.EMBEDDING_STATS)
        return stats
    except Exception as e:
        logger.error(f"Get embedding stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/few-shot/stats")
async def get_few_shot_stats():
    """
    Get statistics about the few-shot learning system.
    
    Returns information about available domains, examples, and transfer rules.
    """
    service = get_few_shot_service()
    return service.get_stats()


@router.get("/few-shot/domains")
async def detect_domains(
    question: str = Query(..., description="Question to analyze"),
    service_name: Optional[str] = Query(None, description="Service name for context"),
):
    """
    Detect relevant log domains for a question.
    
    Uses keyword analysis to determine which log domains are most relevant
    for answering the user's question.
    """
    service = get_few_shot_service()
    domains = service.detect_domains(question, service_name or "")
    
    return {
        "question": question,
        "detected_domains": [
            {
                "domain": domain,
                "confidence": round(score, 3),
                "description": service.get_domain_description(domain)
            }
            for domain, score in domains[:5]
        ]
    }


@router.get("/few-shot/examples")
async def get_relevant_examples(
    question: str = Query(..., description="Question to find examples for"),
    service_name: Optional[str] = Query(None, description="Service name for context"),
    max_examples: int = Query(3, description="Maximum examples to return"),
):
    """
    Get relevant few-shot examples for a question.
    
    Returns examples from the training set that are most relevant to the
    user's question, including cross-domain examples that may apply.
    """
    service = get_few_shot_service()
    
    # Detect domains first
    domains = service.detect_domains(question, service_name or "")
    
    # Get relevant examples
    examples = service.get_relevant_examples(domains, question, max_examples)
    
    # Get transfer rules
    transfer_rules = service.get_transfer_rules(domains)
    
    return {
        "question": question,
        "detected_domains": [d[0] for d in domains[:3]],
        "examples": [
            {
                "id": ex.get("id"),
                "source_domain": ex.get("source_domain"),
                "question": ex.get("question"),
                "applicable_domains": ex.get("applicable_domains", []),
                "transferable_patterns": ex.get("transferable_patterns", []),
            }
            for ex in examples
        ],
        "transfer_rules": transfer_rules[:5],
    }
