"""Monitor profile tests for Secure Agentic Runtime components."""

from __future__ import annotations

import json

from opensandbox_server.sar.monitor import (
    ALWAYS_BLOCKED_SYSCALLS,
    build_monitor_profile,
    generate_seccomp_json,
    risk_from_score,
)
from opensandbox_server.sar.schemas import RiskLevel, SeccompAction, ToolType


class TestMonitor:
    """Tests for the monitor profile builder."""

    def test_build_low_risk_profile(self):
        profile = build_monitor_profile(ToolType.PYTHON, RiskLevel.LOW)
        assert profile.no_new_privileges is True
        assert profile.block_ptrace is True
        assert profile.max_processes == 64

    def test_build_high_risk_profile_restricted(self):
        profile = build_monitor_profile(ToolType.SHELL, RiskLevel.HIGH)
        assert profile.max_processes <= 32
        assert profile.max_open_files <= 128
        assert profile.kill_on_violation is True

    def test_build_critical_risk_profile(self):
        profile = build_monitor_profile(ToolType.SHELL, RiskLevel.CRITICAL)
        assert profile.max_processes <= 16
        assert profile.max_open_files <= 64
        audit_rules = [rule for rule in profile.seccomp_rules if rule.action == SeccompAction.LOG]
        assert len(audit_rules) > 0

    def test_risk_from_score(self):
        assert risk_from_score(0.1) == RiskLevel.LOW
        assert risk_from_score(0.4) == RiskLevel.MEDIUM
        assert risk_from_score(0.7) == RiskLevel.HIGH
        assert risk_from_score(0.9) == RiskLevel.CRITICAL

    def test_generate_seccomp_json(self):
        profile = build_monitor_profile(ToolType.PYTHON, RiskLevel.MEDIUM)
        seccomp_json = generate_seccomp_json(profile)
        parsed = json.loads(seccomp_json)
        assert parsed["defaultAction"] == "SCMP_ACT_ALLOW"
        assert len(parsed["syscalls"]) > 0
        ptrace_rules = [rule for rule in parsed["syscalls"] if "ptrace" in rule.get("names", [])]
        assert len(ptrace_rules) > 0

    def test_always_blocked_syscalls_present(self):
        assert "kexec_load" in ALWAYS_BLOCKED_SYSCALLS

        profile = build_monitor_profile(ToolType.PYTHON, RiskLevel.LOW)
        all_blocked = set()
        for rule in profile.seccomp_rules:
            if rule.action in (SeccompAction.ERRNO, SeccompAction.KILL):
                all_blocked.update(rule.syscalls)

        assert "kexec_load" in all_blocked
        assert "reboot" in all_blocked
