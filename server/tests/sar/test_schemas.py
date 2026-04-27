"""Pydantic schema tests for Secure Agentic Runtime components."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from opensandbox_server.sar.schemas import (
    ApprovedCapabilities,
    CapabilityPermissions,
    CapabilityToken,
    ExecutionStatus,
    GatewayResponse,
    JudgeDecision,
    JudgeVerdict,
    MountProjection,
    NetworkMode,
    ToolCallRequest,
    ToolCallSpec,
    ToolType,
)


class TestSchemas:
    """Tests for Pydantic schema models."""

    def test_tool_call_request_defaults(self):
        req = ToolCallRequest(
            user_goal="Summarize report.pdf",
            tool_call=ToolCallSpec(
                tool=ToolType.PYTHON,
                action="run_script",
                args={"script": "print('hi')"},
            ),
        )
        assert req.task_id
        assert req.user_goal == "Summarize report.pdf"
        assert req.tool_call.tool == ToolType.PYTHON
        assert req.requested_resources.files == []
        assert req.requested_resources.network == []

    def test_capability_token_fields(self):
        now = datetime.now(timezone.utc)
        token = CapabilityToken(
            task_id="task_123",
            permissions=CapabilityPermissions(
                read_files=["/input/report.pdf"],
                max_runtime_seconds=30,
            ),
            expires_at=now + timedelta(seconds=30),
        )
        assert token.capability_id.startswith("cap_")
        assert token.task_id == "task_123"
        assert not token.revoked
        assert token.permissions.read_files == ["/input/report.pdf"]
        assert token.permissions.network_mode == NetworkMode.OFF

    def test_judge_decision_model(self):
        decision = JudgeDecision(
            verdict=JudgeVerdict.ALLOW,
            reason="Consistent with user goal.",
            risk_score=0.22,
            approved_capabilities=ApprovedCapabilities(
                files_read=["report.pdf"],
                ttl_seconds=60,
            ),
        )
        assert decision.verdict == JudgeVerdict.ALLOW
        assert decision.risk_score == 0.22
        assert decision.approved_capabilities.files_read == ["report.pdf"]

    def test_mount_projection(self):
        mount = MountProjection(
            host_path="/Users/me/Documents/report.pdf",
            sandbox_path="/input/report.pdf",
            read_only=True,
        )
        assert mount.read_only is True

    def test_gateway_response(self):
        resp = GatewayResponse(
            task_id="task_1",
            status=ExecutionStatus.DENIED,
            judge_decision=JudgeDecision(
                verdict=JudgeVerdict.DENY,
                reason="blocked",
                risk_score=1.0,
            ),
            error="blocked",
        )
        assert resp.status == ExecutionStatus.DENIED
        assert resp.execution_result is None
