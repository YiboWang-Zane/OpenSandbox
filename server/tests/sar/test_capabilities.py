"""Capability token manager tests for Secure Agentic Runtime components."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from opensandbox_server.sar.schemas import ApprovedCapabilities, NetworkMode


class TestCapabilities:
    """Tests for the capability token manager."""

    def setup_method(self):
        from opensandbox_server.sar.capabilities import CapabilityManager

        self.mgr = CapabilityManager(signing_key="test-key-123", max_ttl_seconds=300)

    def _make_approved(self, **overrides) -> ApprovedCapabilities:
        defaults = {
            "files_read": ["/input/report.pdf"],
            "files_write": ["/output/summary.txt"],
            "network": [],
            "env_vars": [],
            "ttl_seconds": 60,
        }
        defaults.update(overrides)
        return ApprovedCapabilities(**defaults)

    def test_mint_and_validate(self):
        approved = self._make_approved()
        token = self.mgr.mint("task_1", approved)
        assert token.task_id == "task_1"
        assert not token.revoked
        assert token.permissions.read_files == ["/input/report.pdf"]

        validated = self.mgr.validate(token.capability_id)
        assert validated is not None
        assert validated.capability_id == token.capability_id

    def test_ttl_clamped_to_max(self):
        approved = self._make_approved(ttl_seconds=3600)
        token = self.mgr.mint("task_1", approved)
        assert token.permissions.max_runtime_seconds == 300

    def test_revoke(self):
        approved = self._make_approved()
        token = self.mgr.mint("task_1", approved)

        assert self.mgr.revoke(token.capability_id) is True
        assert self.mgr.validate(token.capability_id) is None

    def test_revoke_nonexistent(self):
        assert self.mgr.revoke("cap_nonexistent") is False

    def test_revoke_for_task(self):
        approved = self._make_approved()
        t1 = self.mgr.mint("task_A", approved)
        t2 = self.mgr.mint("task_A", approved)
        t3 = self.mgr.mint("task_B", approved)

        count = self.mgr.revoke_for_task("task_A")
        assert count == 2
        assert self.mgr.validate(t1.capability_id) is None
        assert self.mgr.validate(t2.capability_id) is None
        assert self.mgr.validate(t3.capability_id) is not None

    def test_sign_and_verify(self):
        approved = self._make_approved()
        token = self.mgr.mint("task_1", approved)
        sig = self.mgr.sign_token(token)
        assert self.mgr.verify_signature(token, sig) is True
        assert self.mgr.verify_signature(token, "tampered") is False

    def test_expired_token_invalid(self):
        approved = self._make_approved(ttl_seconds=1)
        token = self.mgr.mint("task_1", approved)

        token.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        self.mgr._store.store(token)

        assert self.mgr.validate(token.capability_id) is None

    def test_cleanup_expired(self):
        approved = self._make_approved(ttl_seconds=1)
        token = self.mgr.mint("task_1", approved)
        token.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        self.mgr._store.store(token)

        removed = self.mgr.cleanup()
        assert removed == 1

    def test_network_mode_set(self):
        approved = self._make_approved(network=["api.example.com"])
        token = self.mgr.mint("task_1", approved, network_mode=NetworkMode.ALLOWLIST_ONLY)
        assert token.permissions.network_mode == NetworkMode.ALLOWLIST_ONLY
        assert token.permissions.network_allowlist == ["api.example.com"]
