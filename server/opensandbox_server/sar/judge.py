"""
Semantic judge layer for the Secure Agentic Runtime

Validates the intent of an LLM-generated tool call against the original
user mission before execution is allowed. The judge answers:

 1. Is the action relevant to the user's stated goal?
 2. Is the requested file/network access justified by the goal?
 3. Is the tool choice proportional?
 4. Is the command attempting to access secrets, system paths, or
    unrelated sensitive resources?

The judge is deterministic by default (rule-based). An optional LLM-backed
judge can be enabled for deeper semantic checks, but the rule engine
always runs first as a hard guardrail.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from typing import List, Optional, Set

from opensandbox_server.sar.schemas import (
    ApprovedCapabilities,
    JudgeDecision,
    JudgeVerdict,
    NetworkMode,
    RiskLevel,
    ToolCallRequest,
    ToolType,
)

logger = logging.getLogger(__name__)


# Deny-listed path patterns
SENSITIVE_PATH_PATTERNS: List[str] = [
    # SSH keys and config
    "*/.ssh/*",
    "*/id_rsa*",
    "*/id_ed25519*",
    "*/id_ecdsa*",
    "*/authorized_keys",
    "*/known_hosts",
    # Cloud credentials
    "*/.aws/*",
    "*/.azure/*",
    "*/.gcp/*",
    "*/.config/gcloud/*",
    "*/.kube/*",
    # Docker / container runtime
    "*/docker.sock",
    "*/.docker/*",
    "/var/run/docker.sock",
    "/var/run/containerd/*",
    # System paths
    "/etc/shadow",
    "/etc/passwd",
    "/etc/sudoers*",
    "/proc/*/mem",
    "/proc/kcore",
    "/sys/firmware/*",
    "/dev/mem",
    "/dev/kmem",
    # Environment / secrets
    "*/.env",
    "*/.env.*",
    "*/.netrc",
    "*/.npmrc",
    "*/.pypirc",
    "*/credentials",
    "*/credentials.json",
    "*/service-account*.json",
    "*/.git-credentials",
    # Private keys
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
]

# Tool types that carry higher inherent risk
HIGH_RISK_TOOLS: Set[ToolType] = {ToolType.SHELL, ToolType.NETWORK, ToolType.DATABASE}

# Base risk scores by tool type
TOOL_BASE_RISK: dict[ToolType, float] = {
    ToolType.PYTHON: 0.3,
    ToolType.SHELL: 0.6,
    ToolType.FILESYSTEM: 0.2,
    ToolType.NETWORK: 0.5,
    ToolType.BROWSER: 0.4,
    ToolType.DATABASE: 0.5,
    ToolType.CUSTOM: 0.4,
}

# Default maximum TTL granted by judge
DEFAULT_TTL_SECONDS = 60

# Maximum files the judge will approve in a single call
MAX_APPROVED_FILES = 50


def _matches_sensitive_pattern(path: str) -> bool:
    """Check if a path matches any sensitive pattern."""
    normalized = path.replace("\\", "/")
    return any(fnmatch.fnmatch(normalized, pat) for pat in SENSITIVE_PATH_PATTERNS)


def _extract_paths_from_args(args: dict) -> List[str]:
    """Heuristically extract file paths from tool call arguments."""
    paths: List[str] = []
    for key, value in args.items():
        if isinstance(value, str) and ("/" in value or "\\" in value):
            # Check if this looks like a path
            if re.match(r"^[/~.]", value) or re.match(r"^[A-Za-z]:[/\\]", value):
                paths.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and ("/" in item or "\\" in item):
                    paths.append(item)
    return paths


def _contains_shell_injection_risk(args: dict) -> bool:
    """Detect common shell injection patterns in arguments."""
    dangerous_patterns = [
        r";\s*rm\s",
        r"\|\s*bash",
        r"\|\s*sh\b",
        r"`[^`]+`",
        r"\$\([^)]+\)",
        r">\s*/dev/",
        r">\s*/etc/",
        r">\s*/proc/",
        r"curl\s+.*\|\s*(bash|sh)",
        r"wget\s+.*\|\s*(bash|sh)",
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__",
        r"os\.system",
        r"subprocess",
        r"shutil\.rmtree",
    ]
    combined = re.compile("|".join(dangerous_patterns), re.IGNORECASE)
    for value in args.values():
        if isinstance(value, str) and combined.search(value):
            return True
    return False


class Judge:
    """
    Deterministic rule-based judge for tool call validation.

    Evaluates every tool call request against hard safety rules and
    heuristic relevance checks. Returns a structured verdict with
    risk score and scoped capabilities.
    """

    def __init__(
        self,
        *,
        default_ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_approved_files: int = MAX_APPROVED_FILES,
        extra_sensitive_patterns: Optional[List[str]] = None,
        allowed_output_prefixes: Optional[List[str]] = None,
    ) -> None:
        self._default_ttl = default_ttl_seconds
        self._max_files = max_approved_files
        self._sensitive_patterns = list(SENSITIVE_PATH_PATTERNS)
        if extra_sensitive_patterns:
            self._sensitive_patterns.extend(extra_sensitive_patterns)
        self._allowed_output_prefixes = allowed_output_prefixes or ["/workspace/output/", "/output/", "/tmp/"]

    def evaluate(self, request: ToolCallRequest) -> JudgeDecision:
        """
        Evaluate a tool call request and return a judge decision.

        The evaluation pipeline:
        1. Hard deny checks (sensitive paths, injection patterns)
        2. Resource scope validation
        3. Tool-goal relevance heuristic
        4. Risk score computation
        5. Capability scoping
        """
        tool = request.tool_call.tool
        action = request.tool_call.action
        args = request.tool_call.args
        requested = request.requested_resources
        user_goal = request.user_goal.lower()

        # Collect all referenced paths
        all_paths = list(requested.files) + _extract_paths_from_args(args)

        # Hard deny: sensitive path access
        for path in all_paths:
            if _matches_sensitive_pattern(path):
                logger.warning(
                    "DENY task=%s: sensitive path access attempted: %s",
                    request.task_id, path,
                )
                return JudgeDecision(
                    verdict=JudgeVerdict.DENY,
                    reason=f"Access to sensitive path '{path}' is blocked regardless of user goal.",
                    risk_score=1.0,
                    denied_resources=[path],
                )

        # Hard deny: shell injection patterns
        if _contains_shell_injection_risk(args):
            logger.warning(
                "DENY task=%s: shell injection risk detected in arguments",
                request.task_id,
            )
            return JudgeDecision(
                verdict=JudgeVerdict.DENY,
                reason="Tool call arguments contain patterns associated with command injection.",
                risk_score=0.95,
            )

        # Hard deny: script content referencing sensitive paths
        script_content = args.get("script", "") or args.get("code", "") or args.get("command", "")
        if isinstance(script_content, str):
            for pattern in self._sensitive_patterns:
                # Convert glob to a simpler check for script content
                literal = pattern.replace("*", "").replace("?", "")
                if literal and len(literal) > 3 and literal in script_content:
                    logger.warning(
                        "DENY task=%s: script references sensitive path pattern: %s",
                        request.task_id, pattern,
                    )
                    return JudgeDecision(
                        verdict=JudgeVerdict.DENY,
                        reason=f"Script content references sensitive resource matching '{pattern}'.",
                        risk_score=0.9,
                        denied_resources=[pattern],
                    )

        # Risk scoring
        risk = TOOL_BASE_RISK.get(tool, 0.4)

        # Adjust for network access
        if requested.network:
            risk = min(risk + 0.15, 1.0)

        # Adjust for number of files
        if len(all_paths) > 10:
            risk = min(risk + 0.1, 1.0)

        # Adjust for env var access
        if requested.env_vars:
            risk = min(risk + 0.1, 1.0)

        # Relevance check: does the tool make sense for the goal?
        # This is a heuristic; the LLM-backed judge can do deeper checks.
        relevance_issue = self._check_relevance(tool, action, user_goal, requested)
        if relevance_issue:
            # Escalate rather than hard-deny for relevance issues
            logger.info(
                "ESCALATE task=%s: relevance concern: %s",
                request.task_id, relevance_issue,
            )
            return JudgeDecision(
                verdict=JudgeVerdict.ESCALATE,
                reason=relevance_issue,
                risk_score=min(risk + 0.2, 1.0),
            )

        # Scope capabilities
        approved_read: List[str] = []
        approved_write: List[str] = []

        for path in all_paths:
            # Determine if this is a read or write path
            is_output = any(path.startswith(prefix) for prefix in self._allowed_output_prefixes)
            if is_output:
                approved_write.append(path)
            else:
                approved_read.append(path)

        # Cap file counts
        if len(approved_read) > self._max_files:
            return JudgeDecision(
                verdict=JudgeVerdict.DENY,
                reason=f"Too many files requested ({len(approved_read)} > {self._max_files}).",
                risk_score=0.8,
            )

        # Determine network mode
        network_mode_for_task = NetworkMode.OFF
        approved_network: List[str] = []
        if requested.network:
            network_mode_for_task = NetworkMode.ALLOWLIST_ONLY
            approved_network = list(requested.network)

        # Compute TTL based on risk
        ttl = self._default_ttl
        if risk > 0.6:
            ttl = max(ttl // 2, 10)  # Reduce TTL for high-risk calls
        elif risk < 0.3:
            ttl = min(ttl * 2, 300)  # Allow more time for low-risk calls

        approved = ApprovedCapabilities(
            files_read=approved_read,
            files_write=approved_write,
            network=approved_network,
            env_vars=list(requested.env_vars),
            ttl_seconds=ttl,
        )

        logger.info(
            "ALLOW task=%s tool=%s risk=%.2f ttl=%ds read=%d write=%d net=%d",
            request.task_id, tool.value, risk, ttl,
            len(approved_read), len(approved_write), len(approved_network),
        )

        return JudgeDecision(
            verdict=JudgeVerdict.ALLOW,
            reason=f"{tool.value}/{action} is consistent with the stated goal.",
            risk_score=round(risk, 2),
            approved_capabilities=approved,
        )

    def _check_relevance(
        self,
        tool: ToolType,
        action: str,
        user_goal: str,
        requested: "RequestedResources",
    ) -> Optional[str]:
        """
        Heuristic relevance check between tool call and user goal.

        Returns a concern string if the tool call seems misaligned,
        None if it appears relevant.
        """
        from opensandbox_server.sar.schemas import RequestedResources

        # Shell access for a pure reading/summarization goal
        if tool == ToolType.SHELL and any(
            keyword in user_goal
            for keyword in ("summarize", "summary", "read", "analyze", "review", "describe")
        ):
            if action not in ("cat", "head", "tail", "wc", "grep"):
                return (
                    f"Shell tool with action '{action}' may be disproportionate "
                    f"for a goal that appears to be read/analysis oriented."
                )

        # Network access when the goal doesn't mention external resources
        if requested.network and not any(
            keyword in user_goal
            for keyword in (
                "fetch", "download", "api", "web", "http", "url",
                "weather", "search", "browse", "request", "webhook",
                "install", "package", "pip", "npm",
            )
        ):
            return (
                "Network access requested but the user goal does not appear "
                "to require external resources."
            )

        # Database access when goal doesn't mention data/db
        if tool == ToolType.DATABASE and not any(
            keyword in user_goal
            for keyword in ("database", "db", "query", "sql", "data", "table", "record")
        ):
            return (
                "Database tool requested but the user goal does not appear "
                "to involve database operations."
            )

        return None