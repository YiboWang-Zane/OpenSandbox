"""
Resource projection for the Secure Agentic Runtime

Converts capability token permissions into concrete mount bindings and
environment injection for an ephemeral runtime. The host filesystem is
never exposed directly; instead, only approved files are mounted at
synthetic in-sandbox paths.

Principles:
- Host paths are hidden from the sandbox process.
- Read-only by default; write only to explicitly approved output paths.
- No home directory, .ssh, cloud creds, Docker socket, or /var/run.
- Secret material is excluded unless explicitly approved.
"""

from __future__ import annotations

import logging
import os
import posixpath
from typing import Dict, List, Optional

from opensandbox_server.sar.schemas import (
    CapabilityToken,
    MountProjection,
    NetworkMode,
    ResourceProjectionPlan,
)

logger = logging.getLogger(__name__)

# In-sandbox mount prefixes
INPUT_PREFIX = "/input"
OUTPUT_PREFIX = "/output"
WORKSPACE_DIR = "/workspace"


class ResourceProjector:
    """
    Converts capability permissions into a concrete resource projection plan.

    The projector creates mount mappings that hide host paths and enforces
    the principle of least privilege: only files listed in the capability
    token are visible inside the sandbox.
    """

    def __init__(
        self,
        *,
        input_prefix: str = INPUT_PREFIX,
        output_prefix: str = OUTPUT_PREFIX,
        workspace_dir: str = WORKSPACE_DIR,
        blocked_host_prefixes: Optional[List[str]] = None,
    ) -> None:
        self._input_prefix = input_prefix
        self._output_prefix = output_prefix
        self._workspace_dir = workspace_dir
        self._blocked_prefixes = blocked_host_prefixes or [
            os.path.expanduser("~/.ssh"),
            os.path.expanduser("~/.aws"),
            os.path.expanduser("~/.azure"),
            os.path.expanduser("~/.gcp"),
            os.path.expanduser("~/.kube"),
            os.path.expanduser("~/.docker"),
            "/var/run/docker.sock",
            "/var/run/containerd",
            "/proc",
            "/sys",
            "/dev",
        ]

    def project(self, token: CapabilityToken, env_values: Optional[Dict[str, str]] = None) -> ResourceProjectionPlan:
        """
        Build a resource projection plan from a capability token.

        Args:
            token: The validated capability token with approved permissions.
            env_values: Actual values for approved environment variables.

        Returns:
            A ResourceProjectionPlan describing exactly what the runtime should mount.

        Raises:
            ValueError: If a requested host path is in the blocked prefix list.
        """
        mounts: List[MountProjection] = []

        # Process read-only mounts
        for host_path in token.permissions.read_files:
            self._validate_host_path(host_path)
            sandbox_path = self._map_to_sandbox_path(host_path, read_only=True)
            mounts.append(MountProjection(
                host_path=host_path,
                sandbox_path=sandbox_path,
                read_only=True,
            ))

        # Process writable mounts
        for host_path in token.permissions.write_files:
            self._validate_host_path(host_path)
            sandbox_path = self._map_to_sandbox_path(host_path, read_only=False)
            mounts.append(MountProjection(
                host_path=host_path,
                sandbox_path=sandbox_path,
                read_only=False,
            ))

        # Filter environment variables: only inject values for approved vars
        safe_env: Dict[str, str] = {}
        if env_values:
            for var_name in token.permissions.env_vars:
                if var_name in env_values:
                    safe_env[var_name] = env_values[var_name]

        plan = ResourceProjectionPlan(
            mounts=mounts,
            network_mode=token.permissions.network_mode,
            network_allowlist=list(token.permissions.network_allowlist),
            env_vars=safe_env,
            working_dir=self._workspace_dir,
        )

        logger.info(
            "Projected resources for capability %s: %d mounts, network=%s, %d env vars",
            token.capability_id,
            len(mounts),
            plan.network_mode.value,
            len(safe_env),
        )
        return plan

    def _validate_host_path(self, host_path: str) -> None:
        """Ensure the host path is not in a blocked prefix."""
        normalized = os.path.normpath(host_path)
        for blocked in self._blocked_prefixes:
            blocked_norm = os.path.normpath(blocked)
            if normalized == blocked_norm or normalized.startswith(blocked_norm + os.sep):
                raise ValueError(
                    f"Host path '{host_path}' is blocked: falls under restricted prefix '{blocked}'."
                )

    def _map_to_sandbox_path(self, host_path: str, read_only: bool) -> str:
        """
        Map a host path to a synthetic in-sandbox path.

        Read files go under /input/, write files go under /output/.
        The filename is preserved but the directory structure is flattened
        to prevent host path leakage.
        """
        basename = posixpath.basename(host_path)
        if not basename:
            # Directory mount — use the last component
            basename = posixpath.basename(host_path.rstrip("/"))

        if read_only:
            return posixpath.join(self._input_prefix, basename)
        else:
            return posixpath.join(self._output_prefix, basename)

    def to_docker_binds(self, plan: ResourceProjectionPlan) -> List[str]:
        """
        Convert a projection plan to Docker bind mount strings.

        Returns:
            List of Docker-format bind strings like "host:container:ro".
        """
        binds: List[str] = []
        for mount in plan.mounts:
            mode = "ro" if mount.read_only else "rw"
            binds.append(f"{mount.host_path}:{mount.sandbox_path}:{mode}")
        return binds

    def to_k8s_volume_mounts(self, plan: ResourceProjectionPlan) -> List[dict]:
        """
        Convert a projection plan to Kubernetes volumeMount specs.

        Returns a schematic list — the caller is responsible for creating
        the corresponding Volume objects.
        """
        mounts = []
        for i, mount in enumerate(plan.mounts):
            mounts.append({
                "name": f"sar-mount-{i}",
                "mountPath": mount.sandbox_path,
                "readOnly": mount.read_only,
            })
        return mounts