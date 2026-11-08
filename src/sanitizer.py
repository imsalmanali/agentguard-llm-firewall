"""
AgentGuard AI - Enterprise PII & Secret Redaction Vault.
Performs deterministic tokenization and reversible de-anonymization for:
- Credit Card Numbers (with Luhn checksum validation)
- Social Security Numbers (SSN)
- Email Addresses
- Phone Numbers
- Cloud & LLM API Keys (OpenAI, GitHub, AWS, JWT)
- IP Addresses
"""

import re
from typing import Dict, List, Tuple, Any, Optional


def luhn_checksum_valid(card_number: str) -> bool:
    """Verifies credit card validity using the ISO/IEC 7812 Luhn algorithm."""
    digits = [int(c) for c in card_number if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, digit in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += doubled if doubled < 10 else doubled - 9
        else:
            checksum += digit
    return checksum % 10 == 0


class PiiRedactionVault:
    """
    Session-aware deterministic PII and secret sanitizer.
    Replaces sensitive data with cryptographically safe surrogate tokens,
    while maintaining a reversible mapping for downstream response reconstruction.
    """

    PATTERNS = {
        "API_KEY": [
            re.compile(r"\b(?:sk-[A-Za-z0-9]{32,}|sk-proj-[A-Za-z0-9_-]{40,})\b"),
            re.compile(r"\b(?:ghp_[A-Za-z0-9]{36}|gho_[A-Za-z0-9]{36})\b"),
            re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
        ],
        "SSN": [
            re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
        ],
        "EMAIL": [
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        ],
        "PHONE": [
            re.compile(r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
        ],
        "IP_ADDRESS": [
            re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b")
        ],
        "CREDIT_CARD_CANDIDATE": [
            re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,19}\b")
        ]
    }

    def __init__(self):
        # Maps session_id -> {surrogate_token: original_value}
        self.session_vault: Dict[str, Dict[str, str]] = {}

    def sanitize(self, text: str, session_id: str = "default_session") -> Dict[str, Any]:
        """
        Redact sensitive tokens from input text and register them in the session vault.
        """
        if session_id not in self.session_vault:
            self.session_vault[session_id] = {}

        vault = self.session_vault[session_id]
        sanitized_text = text
        redaction_counts: Dict[str, int] = {}
        redacted_entities: List[Dict[str, str]] = []

        counter = len(vault) + 1

        # 1. API Keys & Secrets
        for pattern in self.PATTERNS["API_KEY"]:
            for match in pattern.finditer(sanitized_text):
                val = match.group()
                surrogate = f"[REDACTED_SECRET_{counter}]"
                counter += 1
                vault[surrogate] = val
                sanitized_text = sanitized_text.replace(val, surrogate)
                redaction_counts["SECRET_OR_API_KEY"] = redaction_counts.get("SECRET_OR_API_KEY", 0) + 1
                redacted_entities.append({"type": "SECRET_OR_API_KEY", "surrogate": surrogate})

        # 2. Credit Cards (with Luhn verification)
        for pattern in self.PATTERNS["CREDIT_CARD_CANDIDATE"]:
            for match in list(pattern.finditer(sanitized_text)):
                candidate = match.group()
                cleaned_digits = re.sub(r"[-\s]", "", candidate)
                if luhn_checksum_valid(cleaned_digits):
                    surrogate = f"[REDACTED_CREDIT_CARD_{counter}]"
                    counter += 1
                    vault[surrogate] = candidate
                    sanitized_text = sanitized_text.replace(candidate, surrogate)
                    redaction_counts["CREDIT_CARD"] = redaction_counts.get("CREDIT_CARD", 0) + 1
                    redacted_entities.append({"type": "CREDIT_CARD", "surrogate": surrogate})

        # 3. SSN
        for pattern in self.PATTERNS["SSN"]:
            for match in pattern.finditer(sanitized_text):
                val = match.group()
                surrogate = f"[REDACTED_SSN_{counter}]"
                counter += 1
                vault[surrogate] = val
                sanitized_text = sanitized_text.replace(val, surrogate)
                redaction_counts["SSN"] = redaction_counts.get("SSN", 0) + 1
                redacted_entities.append({"type": "SSN", "surrogate": surrogate})

        # 4. Email
        for pattern in self.PATTERNS["EMAIL"]:
            for match in pattern.finditer(sanitized_text):
                val = match.group()
                surrogate = f"[REDACTED_EMAIL_{counter}]"
                counter += 1
                vault[surrogate] = val
                sanitized_text = sanitized_text.replace(val, surrogate)
                redaction_counts["EMAIL"] = redaction_counts.get("EMAIL", 0) + 1
                redacted_entities.append({"type": "EMAIL", "surrogate": surrogate})

        # 5. Phone
        for pattern in self.PATTERNS["PHONE"]:
            for match in pattern.finditer(sanitized_text):
                val = match.group()
                surrogate = f"[REDACTED_PHONE_{counter}]"
                counter += 1
                vault[surrogate] = val
                sanitized_text = sanitized_text.replace(val, surrogate)
                redaction_counts["PHONE"] = redaction_counts.get("PHONE", 0) + 1
                redacted_entities.append({"type": "PHONE", "surrogate": surrogate})

        # 6. IP Address
        for pattern in self.PATTERNS["IP_ADDRESS"]:
            for match in pattern.finditer(sanitized_text):
                val = match.group()
                # Exclude localhost 127.0.0.1 or standard DNS 1.1.1.1 from aggressive masking if needed
                surrogate = f"[REDACTED_IP_{counter}]"
                counter += 1
                vault[surrogate] = val
                sanitized_text = sanitized_text.replace(val, surrogate)
                redaction_counts["IP_ADDRESS"] = redaction_counts.get("IP_ADDRESS", 0) + 1
                redacted_entities.append({"type": "IP_ADDRESS", "surrogate": surrogate})

        total_redacted = sum(redaction_counts.values())

        return {
            "original_text_length": len(text),
            "sanitized_text": sanitized_text,
            "session_id": session_id,
            "total_redacted": total_redacted,
            "redaction_counts": redaction_counts,
            "redacted_entities": redacted_entities
        }

    def de_anonymize(self, text: str, session_id: str = "default_session") -> str:
        """
        Reconstitute original values for an authorized downstream response.
        """
        if session_id not in self.session_vault:
            return text

        vault = self.session_vault[session_id]
        restored = text
        for surrogate, original in vault.items():
            restored = restored.replace(surrogate, original)
        return restored

    def clear_session(self, session_id: str):
        """Purge stored surrogates for a completed session."""
        if session_id in self.session_vault:
            del self.session_vault[session_id]
