"""
Drain template miner - extracts log templates using Drain algorithm.
"""

import logging
import re
from functools import lru_cache
from typing import List, Optional, Tuple

import xxhash
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from ..core.logging import get_logger

logger = get_logger(__name__)

# Suppress drain3 config warnings (we pass config via dict, not file)
logging.getLogger("drain3.template_miner_config").setLevel(logging.ERROR)
logging.getLogger("drain3.template_miner").setLevel(logging.ERROR)


class DrainMiner:
    """
    Wrapper around Drain3 for log template mining.
    
    Extracts templates with <*> placeholders and parameter values.
    """
    
    def __init__(self, depth: int = 4, sim_th: float = 0.4, max_children: int = 100):
        """
        Initialize Drain miner.
        
        Args:
            depth: Depth of prefix tree
            sim_th: Similarity threshold for clustering
            max_children: Max children per node
        """
        self.config = TemplateMinerConfig()
        # Set drain parameters directly (load() expects a filename, not dict)
        self.config.drain_depth = depth
        self.config.drain_sim_th = sim_th
        self.config.drain_max_children = max_children
        self.config.drain_extra_delimiters = ["_", ":", "=", "/", "\\", "[", "]", "(", ")"]
        self.config.mask_prefix = "<"
        self.config.mask_suffix = ">"
        self.config.profiling_enabled = False
        
        self.miner = TemplateMiner(config=self.config)
        
        # Regex for extracting parameters from matched templates
        self._param_pattern = re.compile(r'<\*>')
    
    def add_log_message(self, message: str) -> Tuple[str, List[str], int]:
        """
        Process a log message and extract template.
        
        Args:
            message: Raw log message
            
        Returns:
            Tuple of (template_text, parameters, cluster_id)
        """
        if not message or not message.strip():
            return ("<*>", [message or ""], 0)
        
        # Process through Drain
        result = self.miner.add_log_message(message)
        
        if result is None:
            # Fallback for edge cases
            return ("<*>", [message], 0)
        
        # Handle both old API (object) and new API (dict)
        if isinstance(result, dict):
            template = result.get("template_mined", message)
            cluster_id = result.get("cluster_id", 0)
        else:
            template = result.get_template()
            cluster_id = result.cluster_id
        
        # Extract parameters by comparing template with original message
        parameters = self._extract_parameters(message, template)
        
        return (template, parameters, cluster_id)
    
    def _extract_parameters(self, message: str, template: str) -> List[str]:
        """
        Extract parameter values from message using template.
        
        Args:
            message: Original log message
            template: Template with <*> placeholders
            
        Returns:
            List of parameter values in order
        """
        if "<*>" not in template:
            return []
        
        # Split template into parts around <*>
        parts = template.split("<*>")
        
        if len(parts) < 2:
            return []
        
        parameters = []
        remaining = message
        
        for i, part in enumerate(parts[:-1]):
            # Find the static part in the remaining message
            if part:
                idx = remaining.find(part)
                if idx != -1:
                    # Skip past this static part
                    remaining = remaining[idx + len(part):]
            
            # Now find the next static part
            next_part = parts[i + 1]
            if next_part:
                idx = remaining.find(next_part)
                if idx != -1:
                    # The parameter is everything before the next static part
                    param = remaining[:idx]
                    parameters.append(param)
                    remaining = remaining[idx:]
                else:
                    # No match found, take rest as parameter
                    parameters.append(remaining)
                    remaining = ""
            else:
                # No more static parts, rest is parameter
                parameters.append(remaining)
                remaining = ""
        
        return parameters
    
    def get_cluster_template(self, cluster_id: int) -> Optional[str]:
        """Get template for a cluster ID."""
        cluster = self.miner.drain.id_to_cluster.get(cluster_id)
        if cluster:
            return cluster.get_template()
        return None
    
    @property
    def cluster_count(self) -> int:
        """Get number of clusters (templates)."""
        return len(self.miner.drain.clusters)


def compute_template_hash(service_name: str, template_text: str) -> int:
    """
    Compute deterministic 64-bit hash for a template.
    
    Args:
        service_name: Service name
        template_text: Template text with placeholders
        
    Returns:
        64-bit signed integer hash (SQLite compatible)
    """
    # Normalize template
    normalized = template_text.lower().strip()
    # Collapse multiple whitespace to single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Create unique string
    unique = f"{service_name.lower()}|{normalized}"
    
    # Compute xxhash64 and convert to signed 64-bit integer for SQLite compatibility
    unsigned_hash = xxhash.xxh64(unique.encode('utf-8')).intdigest()
    # Convert to signed 64-bit (SQLite INTEGER is signed)
    if unsigned_hash >= 2**63:
        return unsigned_hash - 2**64
    return unsigned_hash


# Global miner cache per service
_miners: dict[str, DrainMiner] = {}


def get_miner(service_name: str) -> DrainMiner:
    """
    Get or create a Drain miner for a service.
    
    Args:
        service_name: Service name
        
    Returns:
        DrainMiner instance
    """
    if service_name not in _miners:
        _miners[service_name] = DrainMiner()
    return _miners[service_name]


def mine_template(
    message: str,
    service_name: str,
) -> Tuple[str, List[str], int]:
    """
    Mine template from a log message.
    
    Convenience function that handles miner management.
    
    Args:
        message: Log message
        service_name: Service name for the miner
        
    Returns:
        Tuple of (template_text, parameters, template_hash)
    """
    miner = get_miner(service_name)
    template_text, parameters, _ = miner.add_log_message(message)
    template_hash = compute_template_hash(service_name, template_text)
    
    return (template_text, parameters, template_hash)


def reset_miners():
    """Reset all miners (for testing)."""
    global _miners
    _miners = {}
