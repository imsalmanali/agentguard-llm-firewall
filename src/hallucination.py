"""
AgentGuard AI - Real-Time Factual Grounding & Hallucination Scorer.
Quantifies factual alignment between retrieved reference context documents
and generated LLM/Agent responses using multi-tier lexical containment and n-gram fidelity.
"""

import re
from typing import Dict, Any, List, Set


STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "this", "that", "these", "those", "it", "its", "as", "if"
}


def tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into alphanumeric words."""
    words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def extract_bigrams(tokens: List[str]) -> Set[str]:
    """Extract adjacent token pairs for phrase-level matching."""
    return {f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)}


class HallucinationDetector:
    """
    Evaluates factual grounding index and highlights ungrounded candidate claims.
    """

    def __init__(self, grounding_threshold: float = 0.50):
        self.grounding_threshold = grounding_threshold

    def evaluate_grounding(self, reference_context: str, response_text: str) -> Dict[str, Any]:
        """
        Evaluate how faithfully the generated response is grounded in the reference documents.
        """
        if not reference_context.strip():
            return {
                "grounding_score": 0.0,
                "hallucination_risk": "HIGH",
                "is_grounded": False,
                "summary": "Empty reference context provided. Response cannot be verified.",
                "ungrounded_sentences": [s.strip() for s in re.split(r"[.!?]\s+", response_text) if s.strip()]
            }

        ref_tokens = set(tokenize(reference_context))
        ref_bigrams = extract_bigrams(tokenize(reference_context))

        resp_tokens = tokenize(response_text)
        resp_bigrams = extract_bigrams(resp_tokens)

        if not resp_tokens:
            return {
                "grounding_score": 1.0,
                "hallucination_risk": "LOW",
                "is_grounded": True,
                "summary": "Response contains no substantive factual claims.",
                "ungrounded_sentences": []
            }

        # 1. Unigram Overlap Ratio
        overlap_tokens = [t for t in resp_tokens if t in ref_tokens]
        token_overlap_ratio = len(overlap_tokens) / len(resp_tokens)

        # 2. Bigram Semantic Continuity Ratio
        if resp_bigrams:
            bigram_overlap = [b for b in resp_bigrams if b in ref_bigrams]
            bigram_ratio = len(bigram_overlap) / len(resp_bigrams)
        else:
            bigram_ratio = token_overlap_ratio

        # 3. Blended Grounding Score
        grounding_score = 0.60 * token_overlap_ratio + 0.40 * bigram_ratio
        grounding_score = round(min(1.0, max(0.0, grounding_score)), 4)

        # 4. Sentence-by-Sentence Breakdown
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", response_text) if len(s.strip()) > 5]
        ungrounded_sentences: List[str] = []

        for sent in sentences:
            sent_tokens = tokenize(sent)
            if sent_tokens:
                sent_overlap = sum(1 for t in sent_tokens if t in ref_tokens)
                sent_score = sent_overlap / len(sent_tokens)
                if sent_score < 0.35:
                    ungrounded_sentences.append(sent)

        # Determine Risk Level
        if grounding_score >= 0.75:
            risk_level = "LOW"
            is_grounded = True
        elif grounding_score >= 0.45:
            risk_level = "MEDIUM"
            is_grounded = True
        else:
            risk_level = "HIGH"
            is_grounded = False

        return {
            "grounding_score": grounding_score,
            "token_overlap_ratio": round(token_overlap_ratio, 4),
            "bigram_fidelity_ratio": round(bigram_ratio, 4),
            "hallucination_risk": risk_level,
            "is_grounded": is_grounded,
            "total_sentences_evaluated": len(sentences),
            "ungrounded_sentences_count": len(ungrounded_sentences),
            "ungrounded_sentences": ungrounded_sentences
        }
