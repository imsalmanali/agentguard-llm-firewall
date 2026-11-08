"""
AgentGuard AI - Unified Enterprise Agent Security Gateway Orchestrator.
Combines Prompt Injection Detection, PII Vault Sanitization,
Tool Call Sandboxing, and Factual Grounding into a single high-speed pipeline.
"""

import time
from typing import Dict, Any, Optional
from src.detector import PromptInjectionDetector
from src.sanitizer import PiiRedactionVault
from src.tool_guard import ToolCallGuard
from src.hallucination import HallucinationDetector


class AgentGuardEngine:
    """
    Master enterprise orchestrator for AI Agent security, safety, and governance.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
        grounding_threshold: float = 0.50
    ):
        self.detector = PromptInjectionDetector(model_path=model_path)
        self.pii_vault = PiiRedactionVault()
        self.tool_guard = ToolCallGuard(allowed_tools=allowed_tools)
        self.hallucination_detector = HallucinationDetector(grounding_threshold=grounding_threshold)

    def inspect_prompt(self, text: str) -> Dict[str, Any]:
        """Inspect inbound prompt for adversarial injections, jailbreaks, and leaks."""
        start = time.perf_counter()
        result = self.detector.predict(text)
        latency_ms = (time.perf_counter() - start) * 1000.0
        result["latency_ms"] = round(latency_ms, 2)
        return result

    def sanitize_pii(self, text: str, session_id: str = "default_session") -> Dict[str, Any]:
        """Redact sensitive PII and secrets, storing surrogates in session vault."""
        start = time.perf_counter()
        result = self.pii_vault.sanitize(text, session_id=session_id)
        latency_ms = (time.perf_counter() - start) * 1000.0
        result["latency_ms"] = round(latency_ms, 2)
        return result

    def de_anonymize(self, text: str, session_id: str = "default_session") -> str:
        """Restore original sensitive entities for authorized downstream responses."""
        return self.pii_vault.de_anonymize(text, session_id=session_id)

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Guard agent tool invocation against injections and prohibited commands."""
        start = time.perf_counter()
        result = self.tool_guard.validate_tool_call(tool_name, arguments)
        latency_ms = (time.perf_counter() - start) * 1000.0
        result["latency_ms"] = round(latency_ms, 2)
        return result

    def check_hallucination(self, reference_context: str, response_text: str) -> Dict[str, Any]:
        """Quantify factual alignment of agent responses against reference documents."""
        start = time.perf_counter()
        result = self.hallucination_detector.evaluate_grounding(reference_context, response_text)
        latency_ms = (time.perf_counter() - start) * 1000.0
        result["latency_ms"] = round(latency_ms, 2)
        return result

    def intercept_full_pipeline(
        self,
        prompt: str,
        session_id: str = "default_session",
        tool_call: Optional[Dict[str, Any]] = None,
        reference_context: Optional[str] = None,
        agent_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute full lifecycle guardrail inspection across inbound and outbound signals.
        """
        start = time.perf_counter()

        # Step 1: Prompt Security Firewall
        prompt_res = self.inspect_prompt(prompt)

        # Step 2: Inbound PII Sanitization
        pii_res = self.sanitize_pii(prompt, session_id=session_id)

        # Step 3: Tool Call Validation (if tool called)
        tool_res = None
        if tool_call and "tool_name" in tool_call and "arguments" in tool_call:
            tool_res = self.validate_tool_call(tool_call["tool_name"], tool_call["arguments"])

        # Step 4: Grounding & Hallucination Check (if response & context provided)
        grounding_res = None
        if reference_context is not None and agent_response is not None:
            grounding_res = self.check_hallucination(reference_context, agent_response)

        # Final Gateway Decision
        if prompt_res["action"] == "BLOCK":
            decision = "BLOCKED_INJECTION_DETECTED"
            gateway_status = "REJECTED"
        elif tool_res and not tool_res["is_safe"]:
            decision = "BLOCKED_MALICIOUS_TOOL_CALL"
            gateway_status = "REJECTED"
        elif grounding_res and not grounding_res["is_grounded"]:
            decision = "FLAGGED_UNGROUNDED_RESPONSE"
            gateway_status = "WARNING"
        elif pii_res["total_redacted"] > 0:
            decision = "ALLOWED_WITH_PII_REDACTED"
            gateway_status = "SANITIZED"
        else:
            decision = "ALLOWED_CLEAN"
            gateway_status = "PASSED"

        total_latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "gateway_status": gateway_status,
            "decision": decision,
            "total_latency_ms": round(total_latency_ms, 2),
            "prompt_security": prompt_res,
            "pii_sanitization": pii_res,
            "tool_guard": tool_res,
            "grounding_audit": grounding_res
        }
