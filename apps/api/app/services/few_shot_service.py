"""
Few-Shot Learning Service - Cross-domain pattern recognition for log analysis.

This service implements few-shot learning by:
1. Detecting the domain of the user's question and logs
2. Selecting relevant examples from multiple domains
3. Building prompts that leverage cross-domain knowledge transfer
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..core.logging import get_logger

logger = get_logger(__name__)

# Load examples on module import
_EXAMPLES_PATH = Path(__file__).parent / "few_shot_examples.json"
_EXAMPLES_DATA: Optional[Dict] = None


def _load_examples() -> Dict[str, Any]:
    """Load few-shot examples from JSON file."""
    global _EXAMPLES_DATA
    if _EXAMPLES_DATA is None:
        try:
            with open(_EXAMPLES_PATH, 'r') as f:
                _EXAMPLES_DATA = json.load(f)
            if _EXAMPLES_DATA:
                logger.info(f"Loaded {len(_EXAMPLES_DATA.get('cross_domain_examples', []))} few-shot examples")
        except Exception as e:
            logger.error(f"Failed to load few-shot examples: {e}")
            _EXAMPLES_DATA = {"domains": {}, "cross_domain_examples": [], "domain_transfer_rules": []}
    return _EXAMPLES_DATA or {"domains": {}, "cross_domain_examples": [], "domain_transfer_rules": []}


class FewShotService:
    """Service for cross-domain few-shot learning in log analysis."""
    
    def __init__(self):
        self.data = _load_examples()
        self.domains = self.data.get("domains", {})
        self.examples = self.data.get("cross_domain_examples", [])
        self.transfer_rules = self.data.get("domain_transfer_rules", [])
        self.meta_prompts = self.data.get("meta_learning_prompts", {})
    
    def detect_domains(
        self,
        question: str,
        service_name: str = "",
        log_samples: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Detect relevant domains from question, service name, and log samples.
        
        Returns:
            List of (domain_name, confidence_score) tuples sorted by confidence
        """
        text = f"{question} {service_name} {' '.join(log_samples or [])}".lower()
        
        domain_scores = {}
        
        for domain_name, domain_info in self.domains.items():
            keywords = domain_info.get("keywords", [])
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword.lower() in text:
                    score += 1
                    matched_keywords.append(keyword)
            
            # Normalize by keyword count and boost for multiple matches
            if keywords:
                score = (score / len(keywords)) * (1 + 0.1 * len(matched_keywords))
            
            if score > 0:
                domain_scores[domain_name] = score
        
        # Sort by score descending
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        
        # If no domain detected, return application as default
        if not sorted_domains:
            sorted_domains = [("application", 0.5)]
        
        return sorted_domains
    
    def get_relevant_examples(
        self,
        detected_domains: List[Tuple[str, float]],
        question: str,
        max_examples: int = 3
    ) -> List[Dict]:
        """
        Get relevant few-shot examples based on detected domains.
        
        Uses cross-domain transfer: examples from related domains are included
        to help the model understand patterns that transfer across log types.
        """
        relevant_examples = []
        seen_ids = set()
        
        # Get primary domain
        primary_domain = detected_domains[0][0] if detected_domains else "application"
        
        # Score each example by relevance
        example_scores = []
        
        for example in self.examples:
            score = 0
            
            # Primary domain match
            if example.get("source_domain") == primary_domain:
                score += 3
            
            # Applicable domain match
            applicable = example.get("applicable_domains", [])
            for domain, domain_score in detected_domains:
                if domain in applicable:
                    score += 2 * domain_score
            
            # Question similarity (simple keyword matching)
            example_question = example.get("question", "").lower()
            question_lower = question.lower()
            
            # Check for question type match
            question_types = [
                ("why", ["why", "cause", "reason"]),
                ("what", ["what", "which", "describe"]),
                ("how", ["how", "explain"]),
                ("when", ["when", "time", "occurred"]),
                ("unusual", ["unusual", "anomaly", "strange", "weird"]),
                ("attack", ["attack", "security", "breach", "malicious"]),
                ("slow", ["slow", "performance", "latency", "timeout"]),
                ("error", ["error", "fail", "crash", "exception"]),
            ]
            
            for q_type, keywords in question_types:
                q_matches = any(k in question_lower for k in keywords)
                e_matches = any(k in example_question for k in keywords)
                if q_matches and e_matches:
                    score += 1
            
            if score > 0:
                example_scores.append((example, score))
        
        # Sort by score and take top N
        example_scores.sort(key=lambda x: x[1], reverse=True)
        relevant_examples = [ex for ex, _ in example_scores[:max_examples]]
        
        return relevant_examples
    
    def get_transfer_rules(
        self,
        detected_domains: List[Tuple[str, float]]
    ) -> List[str]:
        """Get applicable domain transfer rules."""
        rules = []
        domain_names = [d[0] for d in detected_domains]
        
        for rule in self.transfer_rules:
            source = rule.get("source")
            target = rule.get("target")
            
            # Include rule if source or target domain is detected
            if source in domain_names or target in domain_names or target == "all":
                rules.append(rule.get("pattern", ""))
        
        return rules
    
    def build_few_shot_prompt(
        self,
        question: str,
        service_name: str = "",
        log_samples: Optional[List[str]] = None,
        evidence_context: str = "",
        max_examples: int = 2
    ) -> str:
        """
        Build a complete few-shot prompt with cross-domain knowledge.
        
        This is the main method that combines:
        1. Domain detection
        2. Relevant example selection
        3. Transfer rule application
        4. Context integration
        """
        # Detect domains
        detected_domains = self.detect_domains(question, service_name, log_samples)
        primary_domain = detected_domains[0][0] if detected_domains else "application"
        
        logger.info(f"Detected domains: {detected_domains[:3]}")
        
        # Get relevant examples
        examples = self.get_relevant_examples(detected_domains, question, max_examples)
        
        # Get transfer rules
        transfer_rules = self.get_transfer_rules(detected_domains)
        
        # Build prompt sections
        prompt_parts = []
        
        # 1. Few-shot examples section
        if examples:
            prompt_parts.append("=== EXAMPLE ANALYSES (Learn from these patterns) ===\n")
            for i, ex in enumerate(examples, 1):
                prompt_parts.append(f"**Example {i}** ({ex.get('source_domain', 'general')} domain):")
                prompt_parts.append(f"Question: {ex.get('question', '')}")
                prompt_parts.append(f"Answer:\n{ex.get('answer', '')}")
                
                # Add transferable patterns
                patterns = ex.get("transferable_patterns", [])
                if patterns:
                    prompt_parts.append(f"\nKey Patterns to Remember:")
                    for p in patterns:
                        prompt_parts.append(f"  • {p}")
                prompt_parts.append("")
        
        # 2. Cross-domain transfer rules
        if transfer_rules:
            prompt_parts.append("=== CROSS-DOMAIN INSIGHTS ===")
            prompt_parts.append("Apply these patterns when analyzing:")
            for rule in transfer_rules[:3]:
                prompt_parts.append(f"  • {rule}")
            prompt_parts.append("")
        
        # 3. Current context
        prompt_parts.append("=== CURRENT ANALYSIS TASK ===")
        prompt_parts.append(f"Service: {service_name or 'All Services'}")
        prompt_parts.append(f"Detected Domain(s): {', '.join([d[0] for d in detected_domains[:3]])}")
        prompt_parts.append("")
        
        # 4. Evidence
        if evidence_context:
            prompt_parts.append("=== LOG EVIDENCE ===")
            prompt_parts.append(evidence_context)
            prompt_parts.append("")
        
        # 5. User question
        prompt_parts.append("=== USER QUESTION ===")
        prompt_parts.append(question)
        prompt_parts.append("")
        
        # 6. Instructions
        prompt_parts.append("=== INSTRUCTIONS ===")
        prompt_parts.append("1. Apply patterns learned from the examples above")
        prompt_parts.append("2. Consider cross-domain relationships between log types")
        prompt_parts.append("3. Provide evidence-based analysis with specific log references")
        prompt_parts.append("4. Include actionable recommendations")
        prompt_parts.append("")
        prompt_parts.append("Now provide your analysis:")
        
        return "\n".join(prompt_parts)
    
    def get_domain_description(self, domain_name: str) -> str:
        """Get description for a domain."""
        domain = self.domains.get(domain_name, {})
        return domain.get("description", f"{domain_name} logs")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the few-shot learning data."""
        return {
            "total_domains": len(self.domains),
            "total_examples": len(self.examples),
            "total_transfer_rules": len(self.transfer_rules),
            "domains": list(self.domains.keys()),
            "example_domains": list(set(ex.get("source_domain") for ex in self.examples)),
        }


# Global service instance
_few_shot_service: Optional[FewShotService] = None


def get_few_shot_service() -> FewShotService:
    """Get global few-shot service instance."""
    global _few_shot_service
    if _few_shot_service is None:
        _few_shot_service = FewShotService()
    return _few_shot_service
