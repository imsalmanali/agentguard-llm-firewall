"""
AgentGuard AI - Real-Time Prompt Injection & Adversarial Jailbreak Detector.
Integrates TF-IDF N-Gram Vectorization, Statistical Information Entropy,
Keyword Heuristics, and Cost-Sensitive Machine Learning Classification.
"""

import math
import re
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


# Heuristic patterns for instant short-circuit detection
SIGNATURE_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|prior)\s+(instructions|directives|prompts|rules)",
    r"(?i)disregard\s+(all\s+)?(safety|system|initial)\s+(rules|guidelines|constraints)",
    r"(?i)you\s+are\s+now\s+dan\b",
    r"(?i)bypass\s+(all\s+)?(safety|moderation|content)\s+(filters|checks|rules)",
    r"(?i)output\s+(your\s+)?(system|initial|developer)\s+prompt",
    r"(?i)print\s+(the\s+)?(text\s+above|developer\s+key|internal\s+instructions)",
    r"(?i)drop\s+table\s+\w+",
    r"(?i)rm\s+-rf\s+[/~]",
    r"(?i)powershell\s+(-enc|-encodedcommand)",
    r"(?i)<!--\s*instruction:",
    r"(?i)\[system\s+(alert|notice|message|directive)"
]


def calculate_shannon_entropy(text: str) -> float:
    """Calculates the Shannon Information Entropy H(X) of a text string."""
    if not text:
        return 0.0
    frequencies = {}
    for char in text:
        frequencies[char] = frequencies.get(char, 0) + 1
    length = len(text)
    entropy = 0.0
    for count in frequencies.values():
        p = count / length
        entropy -= p * math.log2(p)
    return float(entropy)


class PromptInjectionDetector:
    """
    Production-grade Prompt Injection & Jailbreak Defense Engine.
    Combines rule-based signatures with a calibrated TF-IDF Machine Learning model.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None
        self.compiled_signatures = [re.compile(p) for p in SIGNATURE_PATTERNS]
        if model_path:
            self.load(model_path)

    def fit(self, texts: list[str], labels: list[int]):
        """
        Train the TF-IDF and Logistic Regression pipeline on labeled prompt sequences.
        """
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=4000,
                sublinear_tf=True,
                lowercase=True
            )),
            ("clf", LogisticRegression(
                C=2.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ])
        self.pipeline.fit(texts, labels)

    def extract_heuristic_features(self, text: str) -> Dict[str, Any]:
        """Extract multi-signal statistical and heuristic indicators."""
        entropy = calculate_shannon_entropy(text)
        length = len(text)
        uppercase_count = sum(1 for c in text if c.isupper())
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())

        matched_signatures = [
            pattern.pattern for pattern in self.compiled_signatures
            if pattern.search(text)
        ]

        return {
            "text_length": length,
            "shannon_entropy": round(entropy, 4),
            "uppercase_ratio": round(uppercase_count / max(1, length), 4),
            "special_char_ratio": round(special_count / max(1, length), 4),
            "signature_matches": matched_signatures,
            "has_signature_match": len(matched_signatures) > 0
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Assess prompt threat severity, probability, and classification.
        """
        heuristics = self.extract_heuristic_features(text)

        # 1. Fast Heuristic Short-Circuit for high-confidence zero-day exploits
        if heuristics["has_signature_match"]:
            threat_prob = 0.992
            is_threat = True
            threat_category = self._categorize_threat(text, heuristics)
            action = "BLOCK"
            risk_level = "CRITICAL"
            return {
                "is_threat": is_threat,
                "threat_probability": threat_prob,
                "risk_level": risk_level,
                "threat_category": threat_category,
                "action": action,
                "detection_mechanism": "HEURISTIC_SIGNATURE_SHORT_CIRCUIT",
                "heuristics": heuristics
            }

        # 2. Machine Learning Inference Pipeline
        if self.pipeline is not None:
            proba = self.pipeline.predict_proba([text])[0]
            threat_prob = float(proba[1])
        else:
            threat_prob = 0.05

        # Determine Risk Level & Action based on calibrated probability thresholds
        if threat_prob >= 0.70:
            risk_level = "CRITICAL"
            action = "BLOCK"
            is_threat = True
        elif threat_prob >= 0.50:
            risk_level = "HIGH"
            action = "BLOCK"
            is_threat = True
        elif threat_prob >= 0.35:
            risk_level = "MEDIUM"
            action = "FLAG_REVIEW"
            is_threat = True
        else:
            risk_level = "CLEAN"
            action = "ALLOW"
            is_threat = False

        threat_category = self._categorize_threat(text, heuristics) if is_threat else "BENIGN"

        return {
            "is_threat": is_threat,
            "threat_probability": round(threat_prob, 4),
            "risk_level": risk_level,
            "threat_category": threat_category,
            "action": action,
            "detection_mechanism": "TF_IDF_LOGISTIC_ENSEMBLE",
            "heuristics": heuristics
        }

    def _categorize_threat(self, text: str, heuristics: Dict[str, Any]) -> str:
        """Classify the attack vector of a detected threat."""
        lower = text.lower()
        if any(kw in lower for kw in ["drop table", "select *", "rm -rf", "curl ", "powershell", "format c:"]):
            return "MALICIOUS_TOOL_HIJACK"
        if any(kw in lower for kw in ["system prompt", "developer key", "instructions you were given", "repeat the text above"]):
            return "SYSTEM_LEAK_PROBE"
        if any(kw in lower for kw in ["system alert", "admin notice", "instruction:", "<!--"]):
            return "PROMPT_INJECTION_INDIRECT"
        if heuristics["shannon_entropy"] > 4.7 or "rot13" in lower or "base64" in lower or "1gn0r3" in lower:
            return "OBFUSCATED_ATTACK"
        if any(kw in lower for kw in ["ignore previous", "disregard", "you are now dan", "bypass"]):
            return "PROMPT_INJECTION_DIRECT"
        return "PROMPT_INJECTION_DIRECT"

    def save(self, filepath: str):
        """Serialize the trained pipeline to disk."""
        joblib.dump(self.pipeline, filepath)

    def load(self, filepath: str):
        """Deserialize a saved pipeline from disk."""
        self.pipeline = joblib.load(filepath)
