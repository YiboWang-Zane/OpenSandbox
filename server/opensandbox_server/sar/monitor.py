"""
Deterministic monitoring and seccomp/process policy for the Secure Agentic Runtime

Builds seccomp profiles and runtime constraints that form the bottom defense
layer. These policies are deterministic (not ML-based) and are enforced at the
kernel level via seccomp-bpf, process limits, and capability restrictions.

V1: seccomp profile generation, process tree limits, privilege escalation blocks.
V2 target: host-side eBPF monitoring for file/network/syscall auditing.
V3 target: policy engine with instant termination on violation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from opensandbox_server.sar.schemas import (
    MonitorProfile,
    RiskLevel,
    SeccompAction,
    SeccompRule,
    ToolType,
)

logger = logging.getLogger(__name__)

# Syscalls that are always blocked in SAR runtimes
ALWAYS_BLOCKED_SYSCALLS: List[str] = [
    "kexec_load",
    "kexec_file_load",
    "reboot",
    "swapon",
    "swapoff",
    "pivot_root",
    "acct",
    "settimeofday",
    "clock_settime",
    "adjtimex",
    "init_module",
    "finit_module",
    "delete_module",
    "create_module",
    "lookup_dcookie",
    "perf_event_open",
    "bpf",
    "userfaultfd",
    "keyctl",
    "request_key",
    "add_key",
    "mbind",
    "move_pages",
    "migrate_pages",
    "set_mempolicy",
    "nfsservctl",
    "vm86",
    "vm86old",
    "modify_ldt",
    "ioperm",
    "iopl",
]

# Syscalls blocked for high-risk tool types
HIGH_RISK_BLOCKED_SYSCALLS: List[str] = [
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "mount",
    "umount2",
    "unshare",
    "setns",
    "clone3",  # Can be used for namespace manipulation
]

# Syscalls that should be logged (V2 eBPF auditing)
AUDIT_SYSCALLS: List[str] = [
    "connect",
    "accept",
    "accept4",
    "bind",
    "listen",
    "execve",
    "execveat",
    "open",
    "openat",
    "openat2",
    "creat",
    "rename",
    "renameat",
    "renameat2",
    "unlink",
    "unlinkat",
    "rmdir",
    "chmod",
    "fchmod",
    "fchmodat",
    "chown",
    "fchown",
    "fchownat",
]


def build_monitor_profile(
    tool_type: ToolType,
    risk_level: RiskLevel = RiskLevel.MEDIUM,
    *,
    max_processes: int = 64,
    max_open_files: int = 256,
    custom_blocked_syscalls: Optional[List[str]] = None,
) -> MonitorProfile:
    """
    Build a monitoring profile appropriate for the tool type and risk level.

    Higher risk levels produce more restrictive profiles with fewer
    allowed syscalls and lower resource limits.
    """
    rules: List[SeccompRule] = []

    # Always block dangerous kernel syscalls
    rules.append(SeccompRule(
        syscalls=ALWAYS_BLOCKED_SYSCALLS,
        action=SeccompAction.ERRNO,
    ))

    # Block high-risk syscalls for medium+ risk
    if risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL):
        rules.append(SeccompRule(
            syscalls=HIGH_RISK_BLOCKED_SYSCALLS,
            action=SeccompAction.ERRNO,
        ))

    # For critical risk, also log common audit syscalls
    if risk_level == RiskLevel.CRITICAL:
        rules.append(SeccompRule(
            syscalls=AUDIT_SYSCALLS,
            action=SeccompAction.LOG,
        ))

    # Custom blocked syscalls
    if custom_blocked_syscalls:
        rules.append(SeccompRule(
            syscalls=custom_blocked_syscalls,
            action=SeccompAction.ERRNO,
        ))

    # Adjust process limits by risk
    if risk_level == RiskLevel.HIGH:
        max_processes = min(max_processes, 32)
        max_open_files = min(max_open_files, 128)
    elif risk_level == RiskLevel.CRITICAL:
        max_processes = min(max_processes, 16)
        max_open_files = min(max_open_files, 64)

    # Shell tools get additionally restricted
    block_raw = True
    if tool_type == ToolType.NETWORK:
        block_raw = True  # Still block raw sockets, but allow regular networking

    profile = MonitorProfile(
        seccomp_rules=rules,
        max_processes=max_processes,
        max_open_files=max_open_files,
        no_new_privileges=True,
        block_ptrace=True,
        block_mount=True,
        block_raw_sockets=block_raw,
        kill_on_violation=(risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)),
    )

    logger.info(
        "Built monitor profile for tool=%s risk=%s: %d rules, max_procs=%d, max_files=%d",
        tool_type.value, risk_level.value, len(rules), max_processes, max_open_files,
    )
    return profile


def risk_from_score(score: float) -> RiskLevel:
    """Convert a numeric risk score [0,1] to a RiskLevel enum."""
    if score >= 0.8:
        return RiskLevel.CRITICAL
    elif score >= 0.6:
        return RiskLevel.HIGH
    elif score >= 0.3:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


def generate_seccomp_json(profile: MonitorProfile) -> str:
    """
    Generate a complete OCI-format seccomp profile JSON string.

    Suitable for passing to Docker's --security-opt seccomp=<profile>.
    """
    syscall_entries: List[Dict[str, Any]] = []

    for rule in profile.seccomp_rules:
        entry: Dict[str, Any] = {
            "names": rule.syscalls,
            "action": rule.action.value,
        }
        syscall_entries.append(entry)

    # Add explicit blocks from boolean flags
    if profile.block_ptrace:
        syscall_entries.append({
            "names": ["ptrace", "process_vm_readv", "process_vm_writev"],
            "action": "SCMP_ACT_ERRNO",
        })

    if profile.block_mount:
        syscall_entries.append({
            "names": ["mount", "umount2", "pivot_root"],
            "action": "SCMP_ACT_ERRNO",
        })

    if profile.block_raw_sockets:
        # Block raw socket creation (type=SOCK_RAW)
        syscall_entries.append({
            "names": ["socket"],
            "action": "SCMP_ACT_ERRNO",
            "args": [
                {"index": 1, "value": 3, "op": "SCMP_CMP_EQ"},  # SOCK_RAW
            ],
        })

    seccomp_profile: Dict[str, Any] = {
        "defaultAction": "SCMP_ACT_ALLOW",
        "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_AARCH64"],
        "syscalls": syscall_entries,
    }

    return json.dumps(seccomp_profile, indent=2)