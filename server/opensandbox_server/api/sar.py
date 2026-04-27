"""
API routes for the Secure Agentic Runtime (SAR) gateway.

These endpoints expose the SAR pipeline over HTTP so that LLM agents
and orchestrators can submit tool call requests for mediated execution.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from opensandbox_server.sar.gateway import GatewayConfig, ToolRequestGateway
from opensandbox_server.sar.schemas import (
    ExecutionStatus,
    GatewayResponse,
    ToolCallRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sar", tags=["Secure Agentic Runtime"])

# Gateway singleton — initialized on first use or at startup via init_sar_gateway().
_gateway: Optional[ToolRequestGateway] = None


def init_sar_gateway(config: Optional[GatewayConfig] = None) -> ToolRequestGateway:
    """Initialize the SAR gateway singleton. Called during app startup if SAR is enabled."""
    global _gateway
    _gateway = ToolRequestGateway(config=config)
    logger.info("SAR gateway initialized (escalation_mode=%s)", (config or GatewayConfig()).escalation_mode)
    return _gateway


def get_gateway() -> ToolRequestGateway:
    """Return the active gateway, raising if SAR is not enabled."""
    if _gateway is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "SAR::NOT_ENABLED",
                "message": "Secure Agentic Runtime is not enabled. Set sar.enabled = true in config.",
            },
        )
    return _gateway


@router.post(
    "/execute",
    response_model=GatewayResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Tool call processed through the SAR pipeline."},
        400: {"description": "Invalid tool call request."},
        403: {"description": "Tool call was denied by the judge."},
        503: {"description": "SAR gateway is not enabled."},
    },
)
async def execute_tool_call(
    request: ToolCallRequest,
) -> GatewayResponse:
    """
    Submit a tool call for mediated execution through the SAR pipeline.

    The tool call is validated, judged against the user's stated goal,
    and if approved, executed in an ephemeral isolated runtime with
    scoped capability tokens.

    Returns the full gateway response including judge decision,
    capability info, and execution result.
    """
    gateway = get_gateway()
    response = await gateway.process(request)

    if response.status == ExecutionStatus.DENIED:
        # Return 403 for denied requests so callers can distinguish
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=response.model_dump(exclude_none=True, mode="json"),
        )

    return response


@router.post(
    "/evaluate",
    response_model=GatewayResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Tool call evaluated (dry-run, no execution)."},
        400: {"description": "Invalid tool call request."},
        503: {"description": "SAR gateway is not enabled."},
    },
)
async def evaluate_tool_call(
    request: ToolCallRequest,
) -> GatewayResponse:
    """
    Evaluate a tool call without executing it (dry-run).

    Runs the validation and judge pipeline and returns the verdict
    without minting capabilities or starting a runtime.
    """
    gateway = get_gateway()
    return await gateway.evaluate_only(request)


@router.post(
    "/terminate/{task_id}",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Task terminated successfully."},
        503: {"description": "SAR gateway is not enabled."},
    },
)
async def terminate_task(task_id: str) -> dict:
    """
    Terminate all runtimes and revoke all capabilities for a task.

    Use this to immediately stop a runaway or compromised execution.
    """
    gateway = get_gateway()
    count = await gateway.terminate_task(task_id)
    return {"task_id": task_id, "terminated_runtimes": count}


@router.post(
    "/cleanup",
    status_code=status.HTTP_200_OK,
)
async def cleanup_expired() -> dict:
    """
    Clean up expired capability tokens.

    Should be called periodically (e.g., via a background task or cron).
    """
    gateway = get_gateway()
    removed = gateway.cleanup()
    return {"expired_tokens_removed": removed}