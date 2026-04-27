"""
Pydantic schemas for the Secure Agentic Runtime (SAR) gateway

Defines the normalized request/response models used across the SAR pipeline:
tool call requests, judge verdicts, capability tokens, and execution results.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Enums
class ToolType(str, Enum):
    """Recognized tool categories for risk classification."""
    PYTHON = "python"
    SHELL = "shell"
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    BROWSER = "browser"
    DATABASE = "database"
    CUSTOM = "custom"


class RiskLevel(str, Enum):
    """Risk classification for a tool call."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NetworkMode(str, Enum):
    """Per-task network access modes."""
    OFF = "off"
    ALLOWLIST_ONLY = "allowlist_only"
    FULL_OUTBOUND = "full_outbound"


class JudgeVerdict(str, Enum):
    """Judge layer decision."""
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"


class ExecutionStatus(str, Enum):
    """Status of a tool call execution."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"
    EXPIRED = "expired"


# Tool call request (from LLM)
class ToolCallSpec(BaseModel):
    """Description of a single tool invocation requested by the LLM."""
    tool: ToolType = Field(..., description="Tool category.")
    action: str = Field(..., description="Specific action within the tool (e.g. 'run_script', 'read_file').")
    args: Dict[str, Any] = Field(default_factory=dict, description="Action-specific arguments.")


class RequestedResources(BaseModel):
    """Resources the tool call claims to need."""
    files: List[str] = Field(default_factory=list, description="File paths the call wants to access.")
    network: List[str] = Field(default_factory=list, description="Network endpoints the call needs.")
    env_vars: List[str] = Field(default_factory=list, description="Environment variable names required.")


class ToolCallRequest(BaseModel):
    """
    Normalized representation of an LLM-generated tool call.

    Every model-generated action is captured in this structure before
    any execution occurs. This is the input to the SAR gateway.
    """
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier.")
    user_goal: str = Field(..., description="Original user intent / mission statement.")
    tool_call: ToolCallSpec = Field(..., description="The tool invocation the LLM wants to perform.")
    requested_resources: RequestedResources = Field(
        default_factory=RequestedResources,
        description="Resources the tool call claims to need.",
    )
    conversation_context: Optional[str] = Field(
        default=None,
        description="Summarized conversation context for the judge layer.",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Judge verdict
class ApprovedCapabilities(BaseModel):
    """Capabilities approved by the judge for this specific execution."""
    files_read: List[str] = Field(default_factory=list, description="Files approved for read access.")
    files_write: List[str] = Field(default_factory=list, description="Files approved for write access.")
    network: List[str] = Field(default_factory=list, description="Network endpoints approved.")
    env_vars: List[str] = Field(default_factory=list, description="Environment variables approved for injection.")
    ttl_seconds: int = Field(default=60, ge=1, le=3600, description="Maximum runtime in seconds.")


class JudgeDecision(BaseModel):
    """
    Result of the semantic judge layer evaluation.

    The judge validates the tool call intent against the user goal,
    checks resource access proportionality, and produces a verdict
    with scoped capabilities if approved.
    """
    verdict: JudgeVerdict = Field(..., description="Allow, deny, or escalate.")
    reason: str = Field(..., description="Human-readable justification for the decision.")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Computed risk score (0=safe, 1=critical).")
    approved_capabilities: Optional[ApprovedCapabilities] = Field(
        default=None,
        description="Scoped capabilities granted if verdict is allow.",
    )
    denied_resources: List[str] = Field(
        default_factory=list,
        description="Resources that were explicitly denied.",
    )


# Capability token
class CapabilityPermissions(BaseModel):
    """Granular permission set for a capability token."""
    read_files: List[str] = Field(default_factory=list)
    write_files: List[str] = Field(default_factory=list)
    network_allowlist: List[str] = Field(default_factory=list)
    env_vars: List[str] = Field(default_factory=list)
    max_runtime_seconds: int = Field(default=60, ge=1, le=3600)
    network_mode: NetworkMode = Field(default=NetworkMode.OFF)


class CapabilityToken(BaseModel):
    """
    Temporal, task-scoped capability token.

    Minted by the gateway after judge approval. Enforced by the
    ephemeral runtime — only resources listed here are available.
    """
    capability_id: str = Field(default_factory=lambda: f"cap_{uuid.uuid4().hex[:12]}")
    task_id: str = Field(..., description="Associated task identifier.")
    permissions: CapabilityPermissions = Field(..., description="Granted permissions.")
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = Field(..., description="Absolute expiration timestamp.")
    revoked: bool = Field(default=False, description="Whether this token has been revoked.")


# Resource projection
class MountProjection(BaseModel):
    """Mapping from host path to in-sandbox path."""
    host_path: str = Field(..., description="Absolute path on the host.")
    sandbox_path: str = Field(..., description="Path visible inside the sandbox (e.g. /input/file.pdf).")
    read_only: bool = Field(default=True, description="Whether the mount is read-only.")


class ResourceProjectionPlan(BaseModel):
    """
    Complete resource projection for an ephemeral runtime.

    Defines exactly which files are visible, what network is available,
    and which env vars are injected. Nothing else is accessible.
    """
    mounts: List[MountProjection] = Field(default_factory=list)
    network_mode: NetworkMode = Field(default=NetworkMode.OFF)
    network_allowlist: List[str] = Field(default_factory=list)
    env_vars: Dict[str, str] = Field(default_factory=dict)
    working_dir: str = Field(default="/workspace")


# Seccomp / monitor profile
class SeccompAction(str, Enum):
    """Seccomp filter actions."""
    ALLOW = "SCMP_ACT_ALLOW"
    ERRNO = "SCMP_ACT_ERRNO"
    KILL = "SCMP_ACT_KILL"
    LOG = "SCMP_ACT_LOG"


class SeccompRule(BaseModel):
    """A single seccomp filter rule."""
    syscalls: List[str] = Field(..., description="Syscall names to match.")
    action: SeccompAction = Field(..., description="Action to apply.")


class MonitorProfile(BaseModel):
    """
    Deterministic monitoring profile applied to an ephemeral runtime.

    Combines seccomp filtering with process limits and kill conditions.
    """
    seccomp_rules: List[SeccompRule] = Field(default_factory=list)
    max_processes: int = Field(default=64, ge=1)
    max_open_files: int = Field(default=256, ge=1)
    no_new_privileges: bool = Field(default=True)
    block_ptrace: bool = Field(default=True)
    block_mount: bool = Field(default=True)
    block_raw_sockets: bool = Field(default=True)
    kill_on_violation: bool = Field(default=True)


# Execution result
class ExecutionResult(BaseModel):
    """
    Result of an executed tool call.

    Returned after the ephemeral runtime completes (or is terminated).
    """
    task_id: str = Field(..., description="Task that was executed.")
    capability_id: str = Field(..., description="Capability token used.")
    status: ExecutionStatus = Field(..., description="Final execution status.")
    exit_code: Optional[int] = Field(default=None, description="Process exit code if applicable.")
    stdout: Optional[str] = Field(default=None, description="Captured stdout (truncated).")
    stderr: Optional[str] = Field(default=None, description="Captured stderr (truncated).")
    output_files: List[str] = Field(default_factory=list, description="Files written to output paths.")
    duration_seconds: Optional[float] = Field(default=None, description="Wall-clock execution time.")
    terminated_reason: Optional[str] = Field(default=None, description="Reason if terminated early.")
    monitor_violations: List[str] = Field(default_factory=list, description="Security violations detected.")


# Gateway response (aggregated)
class GatewayResponse(BaseModel):
    """
    Complete response from the SAR gateway for a tool call request.

    Aggregates the judge decision, capability token, and execution result.
    """
    task_id: str
    status: ExecutionStatus
    judge_decision: JudgeDecision
    capability_token: Optional[CapabilityToken] = None
    execution_result: Optional[ExecutionResult] = None
    error: Optional[str] = None
