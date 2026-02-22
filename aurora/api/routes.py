"""
AURORA API — Route Definitions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from aurora.api.models import (
    UnlearnRequest,
    UnlearnResponse,
    MetricsResponse,
    CertificateResponse,
    HealthResponse,
)
from aurora.config import AuroraConfig
from aurora.pipeline.orchestrator import AuroraPipeline
from aurora.utils.logging import pipeline_logger

logger = pipeline_logger()

router = APIRouter()

# Singleton pipeline instance
_pipeline: Optional[AuroraPipeline] = None


def get_pipeline() -> AuroraPipeline:
    """Get or create the pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = AuroraPipeline()
    return _pipeline


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    pipeline = get_pipeline()
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=hasattr(pipeline, "model"),
    )


@router.post("/unlearn", response_model=UnlearnResponse)
async def unlearn(request: UnlearnRequest):
    """
    Execute the full AURORA unlearning pipeline.

    Accepts a target fact and returns unlearning results with
    certificate and metrics.
    """
    try:
        pipeline = get_pipeline()

        retain = None
        if request.retain_prompts:
            retain = [(p[0], p[1]) for p in request.retain_prompts]

        result = pipeline.run(
            subject=request.subject,
            relation=request.relation,
            obj=request.obj,
            retain_prompts=retain,
            correlated_texts=request.correlated_texts,
        )

        metrics = MetricsResponse(**result.metrics.to_dict())

        cert = None
        if result.certificate:
            cert_dict = result.certificate.to_dict()
            cert = CertificateResponse(
                certificate_id=cert_dict["certificate_id"],
                timestamp=cert_dict["timestamp"],
                root_hash=cert_dict["root_hash"],
                compliance_tag=cert_dict["compliance_tag"],
                target_fact=cert_dict.get("target_fact", {}),
                metrics=cert_dict.get("metrics", {}),
            )

        return UnlearnResponse(
            success=result.success,
            fact=result.fact.to_dict(),
            metrics=metrics,
            certificate=cert,
            graph_stats=result.graph_stats,
            risk_params_count=result.risk_params_count,
            cascade_iterations=result.cascade_iterations,
        )

    except Exception as e:
        logger.error(f"Unlearning failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/certificate/{cert_id}")
async def get_certificate(cert_id: str):
    """
    Retrieve a stored certificate by ID.
    """
    config = AuroraConfig()
    cert_dir = Path(config.certificate_output_dir)

    # Search for certificate file
    cert_file = cert_dir / f"{cert_id}.json"
    if not cert_file.exists():
        # Try with AURORA- prefix
        cert_file = cert_dir / f"AURORA-{cert_id}.json"

    if not cert_file.exists():
        raise HTTPException(status_code=404, detail=f"Certificate {cert_id} not found")

    with open(cert_file) as f:
        return json.load(f)


@router.post("/evaluate")
async def evaluate_model(request: UnlearnRequest):
    """
    Run only the adversarial evaluation suite without unlearning.
    Useful for testing an already-unlearned model.
    """
    try:
        pipeline = get_pipeline()
        if not hasattr(pipeline, "model"):
            pipeline.load_model()

        from aurora.fact_formalization.fact_manager import FactManager
        from aurora.adversarial_suite.evaluator import AdversarialEvaluator

        fm = FactManager()
        forget_set = fm.add_target_fact(
            request.subject, request.relation, request.obj
        )
        retain_set = fm.get_retain_set()

        evaluator = AdversarialEvaluator(pipeline.config)
        metrics, details = evaluator.evaluate(
            pipeline.model, pipeline.tokenizer,
            forget_set, retain_set,
        )

        return {
            "metrics": metrics.to_dict(),
            "details": details,
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
