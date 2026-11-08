"""
AgentGuard AI - Enterprise Production FastAPI Gateway Microservice.
High-throughput REST API for LLM prompt injection defense, PII masking,
tool sandboxing, and factuality grounding verification.
"""

import os
import sys
import json
import time
from typing import Dict, Any

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.engine import AgentGuardEngine
from api.schemas import (
    PromptInspectRequest, PromptInspectResponse,
    PiiSanitizeRequest, PiiSanitizeResponse,
    DeAnonymizeRequest, DeAnonymizeResponse,
    ToolValidationRequest, ToolValidationResponse,
    HallucinationCheckRequest, HallucinationCheckResponse,
    FullInterceptRequest, FullInterceptResponse
)


app = FastAPI(
    title="AgentGuard AI — Enterprise LLM Security Gateway",
    description=(
        "Production-grade, sub-15ms AI Agent Security Gateway & Guardrail Firewall. "
        "Provides real-time prompt injection filtering, session PII redaction vault, "
        "autonomous tool execution sandboxing, and factual grounding evaluation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine singleton
_engine: AgentGuardEngine = None
_startup_time = time.time()


def get_engine() -> AgentGuardEngine:
    global _engine
    if _engine is None:
        model_path = os.path.join(BASE_DIR, "artifacts", "injection_detector.joblib")
        if os.path.exists(model_path):
            _engine = AgentGuardEngine(model_path=model_path)
        else:
            _engine = AgentGuardEngine()
    return _engine


@app.on_event("startup")
def startup_event():
    get_engine()


@app.get("/health", tags=["System Diagnostics"])
def health_check() -> Dict[str, Any]:
    """Inspect microservice uptime, model loaded state, and operational capabilities."""
    uptime_seconds = round(time.time() - _startup_time, 2)
    model_loaded = (
        _engine is not None and
        _engine.detector is not None and
        _engine.detector.pipeline is not None
    )
    return {
        "status": "healthy" if model_loaded else "degraded",
        "service": "AgentGuard AI Security Microservice",
        "version": "1.0.0",
        "uptime_seconds": uptime_seconds,
        "model_loaded": model_loaded,
        "capabilities": [
            "PROMPT_INJECTION_DETECTION",
            "PII_REDACTION_VAULT",
            "REVERSIBLE_DEANONYMIZATION",
            "TOOL_EXECUTION_SANDBOX",
            "SQL_INJECTION_INTERCEPTION",
            "DANGEROUS_OS_COMMAND_BLOCKING",
            "FACTUAL_GROUNDING_EVALUATION"
        ]
    }


@app.post("/v1/guard/inspect", response_model=PromptInspectResponse, tags=["Firewall Core"])
def inspect_prompt_endpoint(payload: PromptInspectRequest):
    """
    Real-time prompt injection, jailbreak, and system prompt exfiltration inspection.
    """
    engine = get_engine()
    try:
        return engine.inspect_prompt(payload.prompt)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prompt inspection failure: {str(e)}"
        )


@app.post("/v1/guard/sanitize-pii", response_model=PiiSanitizeResponse, tags=["Privacy & Compliance"])
def sanitize_pii_endpoint(payload: PiiSanitizeRequest):
    """
    Mask sensitive PII and developer secrets with session-isolated surrogate tokens.
    """
    engine = get_engine()
    try:
        session_id = payload.session_id or "default_session"
        return engine.sanitize_pii(payload.text, session_id=session_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PII sanitization failure: {str(e)}"
        )


@app.post("/v1/guard/de-anonymize", response_model=DeAnonymizeResponse, tags=["Privacy & Compliance"])
def de_anonymize_endpoint(payload: DeAnonymizeRequest):
    """
    Reconstitute original sensitive values from session vault for authorized responses.
    """
    engine = get_engine()
    try:
        restored = engine.de_anonymize(payload.sanitized_text, session_id=payload.session_id)
        return {
            "restored_text": restored,
            "session_id": payload.session_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"De-anonymization failure: {str(e)}"
        )


@app.post("/v1/guard/validate-tool", response_model=ToolValidationResponse, tags=["Agent Execution Guard"])
def validate_tool_endpoint(payload: ToolValidationRequest):
    """
    Intercept autonomous agent tool invocations; blocks SQL injection, destructive commands, and out-of-bounds parameters.
    """
    engine = get_engine()
    try:
        return engine.validate_tool_call(payload.tool_name, payload.arguments)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool validation failure: {str(e)}"
        )


@app.post("/v1/guard/hallucination-check", response_model=HallucinationCheckResponse, tags=["Hallucination & RAG"])
def hallucination_check_endpoint(payload: HallucinationCheckRequest):
    """
    Quantify factual grounding alignment between reference knowledge context and generated agent text.
    """
    engine = get_engine()
    try:
        return engine.check_hallucination(payload.reference_context, payload.response_text)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hallucination check failure: {str(e)}"
        )


@app.post("/v1/guard/full-intercept", response_model=FullInterceptResponse, tags=["Full Lifecycle Gateway"])
def full_intercept_endpoint(payload: FullInterceptRequest):
    """
    Execute full-lifecycle inspection: prompt security, PII sanitization, tool safety, and grounding audit.
    """
    engine = get_engine()
    try:
        return engine.intercept_full_pipeline(
            prompt=payload.prompt,
            session_id=payload.session_id or "default_session",
            tool_call=payload.tool_call,
            reference_context=payload.reference_context,
            agent_response=payload.agent_response
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Full intercept failure: {str(e)}"
        )


@app.get("/v1/guard/metrics", tags=["System Diagnostics"])
def get_benchmark_metrics() -> Dict[str, Any]:
    """Retrieve certified offline benchmark metrics and latency profiles."""
    metrics_path = os.path.join(BASE_DIR, "artifacts", "benchmark_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"message": "Benchmark metrics have not yet been generated. Run src/train.py first."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
