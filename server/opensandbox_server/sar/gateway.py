"""
Tool Request Gateway — the central control point of the Secure Agentic Runtime.

Every LLM-generated tool call must pass through this gateway. The gateway
treats the LLM as untrusted and orchestrates the full SAR pipeline:

1. Receive and validate the tool call request
2. Classify risk
3. Send to the judge layer for intent validation
4. Mint a scoped capability token if approved
5. Build the resource projection plan
6. Build the monitoring profile
7. Dispatch to an ephemeral runtime
8. Collect and return results
9. Revoke the capability and destroy the runtime

No tool call reaches the runtime without passing through this pipeline
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from opensandbox_server.sar.capabilities import CapabilityManager
from opensandbox_server.sar.ephemeral import EphemeralRuntimeConfig, EphemeralRuntimeManager
from opensandbox_server.sar.judge import Judge
from opensandbox_server.sar.monitor import build_monitor_profile, risk_from_score
from opensandbox_server.sar.resource_projection import ResourceProjector
from opensandbox_server.sar.schemas import (
    ExecutionResult,
    ExecutionStatus,
    GatewayResponse,
    JudgeVerdict,
    NetworkMode,
    ToolCallRequest,
    ToolType,
)

logger = logging.getLogger(__name__)

# Valid tool types that the gateway accepts
VALID_TOOL_TYPES = {t.value for t in ToolType}


class GatewayConfig:
    """Configuration for the SAR gateway."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        signing_key: Optional[str] = None,
        max_capability_ttl_seconds: int = 3600,
        default_judge_ttl_seconds: int = 60,
        max_approved_files: int = 50,
        escalation_mode: str = "deny",  # deny or human_review
        extra_sensitive_patterns: Optional[List[str]] = None,
        base_image: str = "python:3.12-slim",
        read_only_rootfs: bool = True,
        max_concurrent_runtimes: int = 10,
    ) -> None:
        self.enabled = enabled
        self.signing_key = signing_key
        self.max_capability_ttl_seconds = max_capability_ttl_seconds
        self.default_judge_ttl_seconds = default_judge_ttl_seconds
        self.max_approved_files = max_approved_files
        self.escalation_mode = escalation_mode
        self.extra_sensitive_patterns = extra_sensitive_patterns
        self.base_image = base_image
        self.read_only_rootfs = read_only_rootfs
        self.max_concurrent_runtimes = max_concurrent_runtimes


class ToolRequestGateway:
    """
    Central execution gateway between the LLM and sandbox runtimes.

    This is the single control point through which every tool call must pass.
    The LLM is treated as untrusted — its outputs are requests, not commands.

    Pipeline:
        validate → judge → mint_capability → project_resources →
        build_monitor → execute_ephemeral → cleanup
    """

    def __init__(self, config: Optional[GatewayConfig] = None) -> None:
        self._config = config or GatewayConfig()

        self._judge = Judge(
            default_ttl_seconds=self._config.default_judge_ttl_seconds,
            max_approved_files=self._config.max_approved_files,
            extra_sensitive_patterns=self._config.extra_sensitive_patterns,
        )

        self._capabilities = CapabilityManager(
            signing_key=self._config.signing_key,
            max_ttl_seconds=self._config.max_capability_ttl_seconds,
        )

        self._projector = ResourceProjector()

        self._runtime_config = EphemeralRuntimeConfig(
            base_image=self._config.base_image,
            read_only_rootfs=self._config.read_only_rootfs,
        )
        self._runtime_manager = EphemeralRuntimeManager(config=self._runtime_config)

        # Public API
    
    async def process(
        self,
        request: ToolCallRequest,
        env_values: Optional[Dict[str, str]] = None,
    ) -> GatewayResponse:
        """
        Process a tool call request through the full SAR pipeline.

        This is the main entry point. Every modeled-generated action
        passes through here before any side effects occur.

        Args:
            request: Normalized tool call request from the LLM.
            env_values: Actual environment variable values for approved vars.

        Returns:
            GatewayResponse with judge decision, capability info, and execution result.
        """
        task_id = request.task_id

        logger.info(
            "Gateway received request task=%s tool=%s action=%s",
            task_id, request.tool_call.tool.value, request.tool_call.action,
        )

        # Validate request structure
        validation_error = self._validate_request(request)
        if validation_error:
            return self._denied_response(task_id, validation_error)

        # Judge the request against user intent
        decision = self._judge.evaluate(request)

        logger.info(
            "Judge verdict for task=%s: %s (risk=%.2f)",
            task_id, decision.verdict.value, decision.risk_score,
        )

        # Handle non-allow verdicts
        if decision.verdict == JudgeVerdict.DENY:
            return GatewayResponse(
                task_id=task_id,
                status=ExecutionStatus.DENIED,
                judge_decision=decision,
                error=f"Denied: {decision.reason}",
            )

        if decision.verdict == JudgeVerdict.ESCALATE:
            if self._config.escalation_mode == "deny":
                return GatewayResponse(
                    task_id=task_id,
                    status=ExecutionStatus.DENIED,
                    judge_decision=decision,
                    error=f"Escalated and auto-denied: {decision.reason}",
                )
            # In "human_review" mode, return escalation status for external handler
            return GatewayResponse(
                task_id=task_id,
                status=ExecutionStatus.PENDING,
                judge_decision=decision,
                error=f"Requires human review: {decision.reason}",
            )

        # Mint capability token
        assert decision.approved_capabilities is not None
        network_mode = self._determine_network_mode(request)

        token = self._capabilities.mint(
            task_id=task_id,
            approved=decision.approved_capabilities,
            network_mode=network_mode,
        )
        signature = self._capabilities.sign_token(token)

        # Build resource projection
        try:
            projection = self._projector.project(token, env_values)
        except ValueError as exc:
            self._capabilities.revoke(token.capability_id)
            return GatewayResponse(
                task_id=task_id,
                status=ExecutionStatus.DENIED,
                judge_decision=decision,
                error=f"Resource projection failed: {exc}",
            )

        # Build monitoring profile
        risk_level = risk_from_score(decision.risk_score)
        monitor = build_monitor_profile(
            tool_type=request.tool_call.tool,
            risk_level=risk_level,
        )

        # Check concurrent runtime limit
        if self._runtime_manager.get_active_count() >= self._config.max_concurrent_runtimes:
            self._capabilities.revoke(token.capability_id)
            return GatewayResponse(
                task_id=task_id,
                status=ExecutionStatus.DENIED,
                judge_decision=decision,
                error="Maximum concurrent runtime limit reached.",
            )

        # Build the execution command from the tool call
        command = self._build_command(request)

        # Execute in ephemeral runtime
        try:
            result = await self._runtime_manager.create_and_run(
                token=token,
                projection=projection,
                monitor=monitor,
                command=command,
            )
        except Exception as exc:
            logger.error("Ephemeral runtime failed for task=%s: %s", task_id, exc)
            result = ExecutionResult(
                task_id=task_id,
                capability_id=token.capability_id,
                status=ExecutionStatus.FAILED,
                terminated_reason=str(exc),
            )
        finally:
            # Always revoke the capability after execution
            self._capabilities.revoke(token.capability_id)

        return GatewayResponse(
            task_id=task_id,
            status=result.status,
            judge_decision=decision,
            capability_token=token,
            execution_result=result,
        )

    async def evaluate_only(self, request: ToolCallRequest) -> GatewayResponse:
        """
        Run only validation and judge evaluation without execution.

        Useful for dry-run checks or UI previews.
        """
        validation_error = self._validate_request(request)
        if validation_error:
            return self._denied_response(request.task_id, validation_error)

        decision = self._judge.evaluate(request)
        status = {
            JudgeVerdict.ALLOW: ExecutionStatus.APPROVED,
            JudgeVerdict.DENY: ExecutionStatus.DENIED,
            JudgeVerdict.ESCALATE: ExecutionStatus.PENDING,
        }.get(decision.verdict, ExecutionStatus.DENIED)

        return GatewayResponse(
            task_id=request.task_id,
            status=status,
            judge_decision=decision,
        )

    async def terminate_task(self, task_id: str) -> int:
        """Terminate all runtimes and revoke all capabilities for a task."""
        cap_count = self._capabilities.revoke_for_task(task_id)
        rt_count = await self._runtime_manager.terminate_task(task_id)
        logger.info(
            "Terminated task %s: %d capabilities revoked, %d runtimes destroyed",
            task_id, cap_count, rt_count,
        )
        return rt_count

    def cleanup(self) -> int:
        """Run periodic cleanup of expired capabilities."""
        return self._capabilities.cleanup()

    # Internal helpers
    def _validate_request(self, request: ToolCallRequest) -> Optional[str]:
        """Validate basic request structure. Returns error message or None."""
        if not request.user_goal or not request.user_goal.strip():
            return "user_goal is required and must not be empty."

        if request.tool_call.tool.value not in VALID_TOOL_TYPES:
            return f"Unknown tool type: {request.tool_call.tool}"

        if not request.tool_call.action or not request.tool_call.action.strip():
            return "tool_call.action is required."

        return None

    def _determine_network_mode(self, request: ToolCallRequest) -> NetworkMode:
        """Determine the appropriate network mode for this request."""
        if not request.requested_resources.network:
            return NetworkMode.OFF

        # Check if any wildcard networks are requested (indicating full outbound)
        for endpoint in request.requested_resources.network:
            if endpoint in ("*", "0.0.0.0/0", "::/0"):
                return NetworkMode.FULL_OUTBOUND

        return NetworkMode.ALLOWLIST_ONLY

    def _build_command(self, request: ToolCallRequest) -> list[str]:
        """
        Build the execution command for the ephemeral runtime.

        Maps tool type + action + args to a concrete command line.
        """
        tool = request.tool_call.tool
        action = request.tool_call.action
        args = request.tool_call.args

        if tool == ToolType.PYTHON:
            script = args.get("script", "")
            if action == "run_script" and script:
                return ["python3", "-c", script]
            filename = args.get("filename", "")
            if action == "run_file" and filename:
                return ["python3", filename]
            return ["python3", "-c", "print('no script provided')"]

        if tool == ToolType.SHELL:
            command = args.get("command", "")
            if command:
                return ["sh", "-c", command]
            return ["echo", "no command provided"]

        if tool == ToolType.FILESYSTEM:
            if action == "read_file":
                path = args.get("path", "")
                return ["cat", path] if path else ["echo", "no path"]
            if action == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                # Write via shell redirection in a safe way
                return ["sh", "-c", f"cat > {path!r} << 'SAREOF'\n{content}\nSAREOF"]
            if action == "list_dir":
                path = args.get("path", "/workspace")
                return ["ls", "-la", path]
            return ["echo", f"unknown filesystem action: {action}"]

        # Default: pass through as a generic command
        cmd = args.get("command", args.get("cmd", ""))
        if cmd:
            return ["sh", "-c", cmd]
        return ["echo", f"unhandled tool type: {tool.value}"]

    def _denied_response(self, task_id: str, error: str) -> GatewayResponse:
        """Build a denied gateway response for validation errors."""
        from opensandbox_server.sar.schemas import JudgeDecision, JudgeVerdict
        return GatewayResponse(
            task_id=task_id,
            status=ExecutionStatus.DENIED,
            judge_decision=JudgeDecision(
                verdict=JudgeVerdict.DENY,
                reason=error,
                risk_score=1.0,
            ),
            error=error,
        )