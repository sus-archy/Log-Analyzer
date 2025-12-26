"""
Vector codec - utilities for encoding/decoding vectors.
"""

import base64
from typing import List

import numpy as np


def encode_vector_b64(vector: np.ndarray) -> str:
    """
    Encode vector to base64 string.
    
    Args:
        vector: NumPy array of floats
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(vector.astype(np.float32).tobytes()).decode("ascii")


def decode_vector_b64(encoded: str) -> np.ndarray:
    """
    Decode base64 string to vector.
    
    Args:
        encoded: Base64 encoded string
        
    Returns:
        NumPy array of float32
    """
    return np.frombuffer(base64.b64decode(encoded), dtype=np.float32)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    L2 normalize a vector for cosine similarity.
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    L2 normalize multiple vectors.
    
    Args:
        vectors: 2D array of vectors (N x D)
        
    Returns:
        Normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return vectors / norms


def list_to_vector(values: List[float]) -> np.ndarray:
    """Convert list of floats to numpy vector."""
    return np.array(values, dtype=np.float32)


def vector_to_list(vector: np.ndarray) -> List[float]:
    """Convert numpy vector to list of floats."""
    return vector.astype(np.float32).tolist()
