"""Gateway pipeline tests for Secure Agentic Runtime components."""

from __future__ import annotations

import pytest

from opensandbox_server.sar.schemas import (
    ExecutionStatus,
    JudgeVerdict,
    RequestedResources,
    ToolCallRequest,
    ToolCallSpec,
    ToolType,
)


class TestGateway:
    """Tests for the tool request gateway pipeline."""

    def setup_method(self):
        from opensandbox_server.sar.gateway import GatewayConfig, ToolRequestGateway

        self.config = GatewayConfig(
            enabled=True,
            signing_key="test-gateway-key",
            escalation_mode="deny",
        )
        self.gateway = ToolRequestGateway(config=self.config)

    def _make_request(self, **kwargs) -> ToolCallRequest:
        defaults = {
            "user_goal": "Summarize financial_report.pdf",
            "tool_call": ToolCallSpec(
                tool=ToolType.PYTHON,
                action="run_script",
                args={"script": "print('summary')"},
            ),
            "requested_resources": RequestedResources(
                files=["financial_report.pdf"],
            ),
        }
        defaults.update(kwargs)
        return ToolCallRequest(**defaults)

    @pytest.mark.asyncio
    async def test_evaluate_only_allow(self):
        req = self._make_request()
        resp = await self.gateway.evaluate_only(req)
        assert resp.status == ExecutionStatus.APPROVED
        assert resp.judge_decision.verdict == JudgeVerdict.ALLOW

    @pytest.mark.asyncio
    async def test_evaluate_only_deny_sensitive_path(self):
        req = self._make_request(
            requested_resources=RequestedResources(
                files=["/home/user/.ssh/id_rsa"],
            ),
        )
        resp = await self.gateway.evaluate_only(req)
        assert resp.status == ExecutionStatus.DENIED
        assert resp.judge_decision.verdict == JudgeVerdict.DENY

    @pytest.mark.asyncio
    async def test_evaluate_only_deny_empty_goal(self):
        req = self._make_request(user_goal="")
        resp = await self.gateway.evaluate_only(req)
        assert resp.status == ExecutionStatus.DENIED

    @pytest.mark.asyncio
    async def test_evaluate_escalation_auto_denied(self):
        """With escalation_mode=deny, escalated requests are auto-denied."""
        req = self._make_request(
            user_goal="Summarize the local report",
            requested_resources=RequestedResources(
                files=["report.pdf"],
                network=["api.suspicious.com"],
            ),
        )
        resp = await self.gateway.evaluate_only(req)
        assert resp.status in (ExecutionStatus.PENDING, ExecutionStatus.DENIED)

    @pytest.mark.asyncio
    async def test_process_deny_does_not_execute(self):
        req = self._make_request(
            requested_resources=RequestedResources(
                files=["/home/user/.ssh/id_rsa"],
            ),
        )
        resp = await self.gateway.process(req)
        assert resp.status == ExecutionStatus.DENIED
        assert resp.execution_result is None
        assert resp.capability_token is None

    @pytest.mark.asyncio
    async def test_terminate_task(self):
        count = await self.gateway.terminate_task("nonexistent_task")
        assert count == 0

    def test_cleanup(self):
        removed = self.gateway.cleanup()
        assert removed == 0

    @pytest.mark.asyncio
    async def test_network_mode_detection_off(self):
        req = self._make_request(
            requested_resources=RequestedResources(files=["report.pdf"]),
        )
        resp = await self.gateway.evaluate_only(req)
        if resp.judge_decision.approved_capabilities:
            assert resp.judge_decision.approved_capabilities.network == []

    @pytest.mark.asyncio
    async def test_network_mode_detection_allowlist(self):
        req = self._make_request(
            user_goal="Fetch data from api.example.com",
            tool_call=ToolCallSpec(
                tool=ToolType.NETWORK,
                action="http_get",
                args={"url": "https://api.example.com/data"},
            ),
            requested_resources=RequestedResources(
                network=["api.example.com"],
            ),
        )
        resp = await self.gateway.evaluate_only(req)
        assert resp.status == ExecutionStatus.APPROVED
        caps = resp.judge_decision.approved_capabilities
        assert caps is not None
        assert "api.example.com" in caps.network
