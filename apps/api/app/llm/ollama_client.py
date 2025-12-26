"""
Ollama client - interfaces with local Ollama for embeddings and chat.
"""

from typing import Any, Dict, List, Optional

import httpx

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class OllamaError(Exception):
    """Error from Ollama API."""
    pass


class OllamaClient:
    """
    Client for Ollama local LLM API.
    
    Provides embeddings and chat completion.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        chat_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 300.0,  # 5 minutes for chat (LLMs can be slow)
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL
            chat_model: Model for chat
            embed_model: Model for embeddings
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.chat_model = chat_model or settings.ollama_chat_model
        self.embed_model = embed_model or settings.ollama_embed_model
        self.api_key = api_key or settings.ollama_api_key
        self.timeout = timeout
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers=headers,
            )
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """List available models."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            model: Model to use (defaults to embed_model)
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            OllamaError: If embedding fails
        """
        model = model or self.embed_model
        
        try:
            client = await self._get_client()
            response = await client.post(
                "/api/embeddings",
                json={
                    "model": model,
                    "prompt": text,
                }
            )
            response.raise_for_status()
            data = response.json()
            
            embedding = data.get("embedding")
            if embedding is None:
                raise OllamaError("No embedding in response")
            
            return embedding
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama embedding HTTP error: {e.response.status_code}")
            raise OllamaError(f"HTTP error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Ollama embedding request error: {e}")
            raise OllamaError(f"Request error: {e}")
        except Exception as e:
            logger.error(f"Ollama embedding error: {e}")
            raise OllamaError(str(e))
    
    async def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            model: Model to use
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            try:
                emb = await self.embed(text, model)
                embeddings.append(emb)
            except OllamaError as e:
                logger.warning(f"Failed to embed text: {e}")
                # Return empty embedding on failure
                embeddings.append([])
        return embeddings
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str:
        """
        Generate chat completion.
        
        Args:
            messages: Chat messages [{"role": "user", "content": "..."}]
            model: Model to use (defaults to chat_model)
            system: System prompt
            temperature: Sampling temperature
            stream: Whether to stream response (not implemented yet)
            
        Returns:
            Generated response text
            
        Raises:
            OllamaError: If chat fails
        """
        model = model or self.chat_model
        
        # Prepend system message if provided
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        try:
            client = await self._get_client()
            response = await client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            
            message = data.get("message", {})
            content = message.get("content", "")
            
            if not content:
                raise OllamaError("No content in response")
            
            return content
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama chat HTTP error: {e.response.status_code}")
            raise OllamaError(f"HTTP error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Ollama chat request error: {e}")
            raise OllamaError(f"Request error: {e}")
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            raise OllamaError(str(e))


# Global client instance
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get global Ollama client instance."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


async def close_ollama_client() -> None:
    """Close global Ollama client."""
    global _ollama_client
    if _ollama_client is not None:
        await _ollama_client.close()
        _ollama_client = None
