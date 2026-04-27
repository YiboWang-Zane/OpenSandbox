"""Resource projection tests for Secure Agentic Runtime components."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from opensandbox_server.sar.schemas import (
    CapabilityPermissions,
    CapabilityToken,
    NetworkMode,
)


class TestResourceProjection:
    """Tests for the resource projection layer."""

    def setup_method(self):
        from opensandbox_server.sar.resource_projection import ResourceProjector

        self.projector = ResourceProjector()

    def _make_token(
        self,
        read_files=None,
        write_files=None,
        network_mode=NetworkMode.OFF,
        env_vars=None,
    ):
        now = datetime.now(timezone.utc)
        return CapabilityToken(
            task_id="task_1",
            permissions=CapabilityPermissions(
                read_files=read_files or [],
                write_files=write_files or [],
                network_allowlist=[],
                env_vars=env_vars or [],
                max_runtime_seconds=60,
                network_mode=network_mode,
            ),
            expires_at=now + timedelta(seconds=60),
        )

    def test_read_file_projected_to_input(self):
        token = self._make_token(read_files=["/Users/me/Documents/report.pdf"])
        plan = self.projector.project(token)
        assert len(plan.mounts) == 1
        mount = plan.mounts[0]
        assert mount.host_path == "/Users/me/Documents/report.pdf"
        assert mount.sandbox_path == "/input/report.pdf"
        assert mount.read_only is True

    def test_write_file_projected_to_output(self):
        token = self._make_token(write_files=["/data/output/summary.txt"])
        plan = self.projector.project(token)
        assert len(plan.mounts) == 1
        mount = plan.mounts[0]
        assert mount.sandbox_path == "/output/summary.txt"
        assert mount.read_only is False

    def test_blocked_ssh_path_raises(self):
        ssh_path = os.path.expanduser("~/.ssh/id_rsa")
        token = self._make_token(read_files=[ssh_path])
        with pytest.raises(ValueError, match="blocked"):
            self.projector.project(token)

    def test_blocked_docker_socket_raises(self):
        token = self._make_token(read_files=["/var/run/docker.sock"])
        with pytest.raises(ValueError, match="blocked"):
            self.projector.project(token)

    def test_env_var_filtering(self):
        token = self._make_token(env_vars=["ALLOWED_VAR"])
        plan = self.projector.project(
            token,
            env_values={"ALLOWED_VAR": "value1", "SECRET_VAR": "should_not_appear"},
        )
        assert plan.env_vars == {"ALLOWED_VAR": "value1"}
        assert "SECRET_VAR" not in plan.env_vars

    def test_network_mode_propagated(self):
        token = self._make_token(network_mode=NetworkMode.ALLOWLIST_ONLY)
        plan = self.projector.project(token)
        assert plan.network_mode == NetworkMode.ALLOWLIST_ONLY

    def test_docker_binds_format(self):
        token = self._make_token(
            read_files=["/data/input.txt"],
            write_files=["/data/output.txt"],
        )
        plan = self.projector.project(token)
        binds = self.projector.to_docker_binds(plan)
        assert len(binds) == 2
        assert any(":ro" in bind for bind in binds)
        assert any(":rw" in bind for bind in binds)

    def test_working_dir_default(self):
        token = self._make_token()
        plan = self.projector.project(token)
        assert plan.working_dir == "/workspace"
