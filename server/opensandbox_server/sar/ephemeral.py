"""
Ephemeral runtime manager for the Secure Agentic Runtime

Manages the lifecycle of zero-state execution environments. Every tool call
executes in a fresh, isolated runtime that is destroyed immediately after
completion. No state carries over between executions

V1 implementation: short-lived Docker containers with read-only base images.
V2 target: Firecracker-backed microVMs with prewarmed snapshots.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from opensandbox_server.sar.schemas import (
    CapabilityToken,
    ExecutionResult,
    ExecutionStatus,
    MonitorProfile,
    ResourceProjectionPlan,
)

logger = logging.getLogger(__name__)


class EphemeralRuntimeConfig:
    """Configuration for ephemeral runtime creation."""

    def __init__(
        self,
        *,
        base_image: str = "python:3.12-slim",
        read_only_rootfs: bool = True,
        tmpfs_size_mb: int = 64,
        memory_limit_mb: int = 256,
        cpu_quota: float = 0.5,
        pids_limit: int = 64,
        auto_remove: bool = True,
    ) -> None:
        self.base_image = base_image
        self.read_only_rootfs = read_only_rootfs
        self.tmpfs_size_mb = tmpfs_size_mb
        self.memory_limit_mb = memory_limit_mb
        self.cpu_quota = cpu_quota
        self.pids_limit = pids_limit
        self.auto_remove = auto_remove


class EphemeralRuntime:
    """
    Represents a single ephemeral execution environment.

    Each instance tracks the lifecycle of one container/microVM that
    was created for a single tool call and will be destroyed after.
    """

    def __init__(
        self,
        *,
        runtime_id: str,
        task_id: str,
        capability_id: str,
        container_id: Optional[str] = None,
    ) -> None:
        self.runtime_id = runtime_id
        self.task_id = task_id
        self.capability_id = capability_id
        self.container_id = container_id
        self.created_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self.status = ExecutionStatus.PENDING
        self.exit_code: Optional[int] = None
        self.stdout: Optional[str] = None
        self.stderr: Optional[str] = None


class EphemeralRuntimeManager:
    """
    Manages ephemeral runtime lifecycle for tool call execution

    Creates a fresh, isolated runtime per tool call:
    - Read-only root filesystem
    - Only approved files mounted via resource projection
    - Network controlled by capability token
    - Destroyed immediately after completion
    - No state carry-over between executions

    This is the V1 implementation using Docker containers
    """

    def __init__(
        self,
        config: Optional[EphemeralRuntimeConfig] = None,
        docker_client: Any = None,
    ) -> None:
        self._config = config or EphemeralRuntimeConfig()
        self._docker = docker_client
        self._active_runtimes: Dict[str, EphemeralRuntime] = {}

    def _get_docker_client(self) -> Any:
        """Lazy-load Docker client."""
        if self._docker is None:
            import docker
            self._docker = docker.from_env()
        return self._docker

    def build_container_config(
        self,
        token: CapabilityToken,
        projection: ResourceProjectionPlan,
        monitor: MonitorProfile,
        command: List[str],
    ) -> Dict[str, Any]:
        """
        Build Docker container creation kwargs from SAR components.

        This produces the container config with:
        - Read-only root filesystem
        - Capability-scoped bind mounts only
        - Enforced resource limits
        - Seccomp profile
        - No privilege escalation
        - Process limits
        - Network isolation
        """
        cfg = self._config

        # Build bind mounts from projection
        binds: List[str] = []
        for mount in projection.mounts:
            mode = "ro" if mount.read_only else "rw"
            binds.append(f"{mount.host_path}:{mount.sandbox_path}:{mode}")

        # Tmpfs for writable areas the process needs (e.g., /tmp)
        tmpfs = {"/tmp": f"size={cfg.tmpfs_size_mb}m,noexec,nosuid,nodev"}

        # Security options
        security_opt: List[str] = []
        if monitor.no_new_privileges:
            security_opt.append("no-new-privileges:true")

        # Dropped capabilities
        cap_drop = ["ALL"]

        # Determine network mode
        from opensandbox_server.sar.schemas import NetworkMode
        network_mode = "none"
        if projection.network_mode == NetworkMode.FULL_OUTBOUND:
            network_mode = "bridge"
        elif projection.network_mode == NetworkMode.ALLOWLIST_ONLY:
            network_mode = "bridge"  # Allowlist enforced by egress sidecar

        # Seccomp profile
        seccomp_profile = self._build_seccomp_profile(monitor)

        container_config: Dict[str, Any] = {
            "image": cfg.base_image,
            "command": command,
            "detach": True,
            "read_only": cfg.read_only_rootfs,
            "network_mode": network_mode,
            "working_dir": projection.working_dir,
            "environment": projection.env_vars,
            "labels": {
                "opensandbox.io/sar": "true",
                "opensandbox.io/task-id": token.task_id,
                "opensandbox.io/capability-id": token.capability_id,
            },
            "host_config": {
                "Binds": binds,
                "Tmpfs": tmpfs,
                "ReadonlyRootfs": cfg.read_only_rootfs,
                "Memory": cfg.memory_limit_mb * 1024 * 1024,
                "NanoCpus": int(cfg.cpu_quota * 1e9),
                "PidsLimit": cfg.pids_limit,
                "CapDrop": cap_drop,
                "SecurityOpt": security_opt,
                "AutoRemove": cfg.auto_remove,
            },
        }

        if seccomp_profile:
            container_config["host_config"]["SecurityOpt"].append(
                f"seccomp={seccomp_profile}"
            )

        return container_config

    async def create_and_run(
        self,
        token: CapabilityToken,
        projection: ResourceProjectionPlan,
        monitor: MonitorProfile,
        command: List[str],
    ) -> ExecutionResult:
        """
        Create an ephemeral runtime, execute the command, and destroy it.

        This is the main entry point for tool call execution. The entire
        lifecycle happens here:
        1. Create container with capability-scoped config
        2. Start container
        3. Wait for completion (with timeout from capability TTL)
        4. Capture output
        5. Destroy container
        6. Return results

        The container is always destroyed, even on failure.
        """
        import uuid
        runtime_id = f"sar-{uuid.uuid4().hex[:12]}"
        runtime = EphemeralRuntime(
            runtime_id=runtime_id,
            task_id=token.task_id,
            capability_id=token.capability_id,
        )
        self._active_runtimes[runtime_id] = runtime

        start_time = time.monotonic()
        violations: List[str] = []

        try:
            client = self._get_docker_client()
            container_config = self.build_container_config(
                token, projection, monitor, command,
            )

            # Create container
            container_name = f"sar-{token.task_id[:8]}-{token.capability_id[-8:]}"
            container = client.containers.create(
                name=container_name,
                **self._flatten_config(container_config),
            )
            runtime.container_id = container.id
            runtime.status = ExecutionStatus.RUNNING
            runtime.started_at = datetime.now(timezone.utc)

            logger.info(
                "Started ephemeral runtime %s (container=%s) for task %s",
                runtime_id, container.short_id, token.task_id,
            )

            # Start and wait with timeout
            container.start()
            timeout = token.permissions.max_runtime_seconds
            result = container.wait(timeout=timeout)

            runtime.exit_code = result.get("StatusCode", -1)
            runtime.status = ExecutionStatus.COMPLETED

            # Capture output (truncated to 64KB)
            max_output = 65536
            try:
                runtime.stdout = container.logs(stdout=True, stderr=False, tail=1000).decode(
                    "utf-8", errors="replace"
                )[:max_output]
                runtime.stderr = container.logs(stdout=False, stderr=True, tail=1000).decode(
                    "utf-8", errors="replace"
                )[:max_output]
            except Exception:
                logger.warning("Failed to capture logs for runtime %s", runtime_id)

        except Exception as exc:
            error_msg = str(exc)
            # Check for timeout
            if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
                runtime.status = ExecutionStatus.EXPIRED
                violations.append(f"Execution exceeded TTL of {token.permissions.max_runtime_seconds}s")
            else:
                runtime.status = ExecutionStatus.FAILED
                violations.append(f"Runtime error: {error_msg}")
            logger.error("Ephemeral runtime %s failed: %s", runtime_id, error_msg)

        finally:
            # Always destroy the container
            runtime.finished_at = datetime.now(timezone.utc)
            self._destroy_runtime(runtime)
            self._active_runtimes.pop(runtime_id, None)

        elapsed = time.monotonic() - start_time

        return ExecutionResult(
            task_id=token.task_id,
            capability_id=token.capability_id,
            status=runtime.status,
            exit_code=runtime.exit_code,
            stdout=runtime.stdout,
            stderr=runtime.stderr,
            output_files=[m.sandbox_path for m in projection.mounts if not m.read_only],
            duration_seconds=round(elapsed, 3),
            terminated_reason=violations[0] if violations else None,
            monitor_violations=violations,
        )

    def _destroy_runtime(self, runtime: EphemeralRuntime) -> None:
        """Force-remove the container, ignoring errors."""
        if runtime.container_id is None:
            return
        try:
            client = self._get_docker_client()
            container = client.containers.get(runtime.container_id)
            container.remove(force=True)
            logger.info(
                "Destroyed ephemeral runtime %s (container=%s)",
                runtime.runtime_id, runtime.container_id[:12],
            )
        except Exception:
            # Container may already be removed (auto_remove=True)
            pass

    def _flatten_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten container_config for the Docker SDK create call."""
        host_config = config.pop("host_config", {})
        # The Docker SDK expects these as top-level kwargs or via host_config
        flat = dict(config)
        if host_config:
            # Map host_config fields to Docker SDK parameter names
            mapping = {
                "Binds": "volumes",
                "ReadonlyRootfs": "read_only",
                "Memory": "mem_limit",
                "NanoCpus": "nano_cpus",
                "PidsLimit": "pids_limit",
                "CapDrop": "cap_drop",
                "SecurityOpt": "security_opt",
                "AutoRemove": "auto_remove",
                "Tmpfs": "tmpfs",
            }
            for docker_key, sdk_key in mapping.items():
                if docker_key in host_config:
                    flat[sdk_key] = host_config[docker_key]
        return flat

    def _build_seccomp_profile(self, monitor: MonitorProfile) -> Optional[str]:
        """
        Build a JSON seccomp profile string from the monitor profile.

        Returns None if no custom rules are defined (use Docker default)
        """
        import json

        if not monitor.seccomp_rules:
            return None

        # Build a seccomp profile in OCI format
        profile: Dict[str, Any] = {
            "defaultAction": "SCMP_ACT_ALLOW",
            "syscalls": [],
        }

        for rule in monitor.seccomp_rules:
            profile["syscalls"].append({
                "names": rule.syscalls,
                "action": rule.action.value,
            })

        # Always block dangerous syscalls if not explicitly configured
        if monitor.block_ptrace:
            profile["syscalls"].append({
                "names": ["ptrace"],
                "action": "SCMP_ACT_ERRNO",
            })
        if monitor.block_mount:
            profile["syscalls"].append({
                "names": ["mount", "umount2"],
                "action": "SCMP_ACT_ERRNO",
            })
        if monitor.block_raw_sockets:
            profile["syscalls"].append({
                "names": ["socket"],
                "action": "SCMP_ACT_ERRNO",
                "args": [{"index": 0, "value": 3, "op": "SCMP_CMP_EQ"}],  # AF_NETLINK raw
            })

        return json.dumps(profile)

    def get_active_count(self) -> int:
        """Return the number of currently active runtimes."""
        return len(self._active_runtimes)

    async def terminate_task(self, task_id: str) -> int:
        """Force-terminate all runtimes for a given task. Returns count terminated."""
        count = 0
        to_destroy = [
            rt for rt in self._active_runtimes.values()
            if rt.task_id == task_id
        ]
        for rt in to_destroy:
            rt.status = ExecutionStatus.TERMINATED
            self._destroy_runtime(rt)
            self._active_runtimes.pop(rt.runtime_id, None)
            count += 1
        if count:
            logger.info("Terminated %d runtimes for task %s", count, task_id)
        return count