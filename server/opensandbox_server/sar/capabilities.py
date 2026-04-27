"""
Capability token management for the Secure Agentic Runtime.

Capability tokens are temporal, task-scoped credentials that encode
exactly what an ephemeral runtime is allowed to do. They are minted
by the gateway after judge approval and enforced by the runtime layer.

Tokens are:
- Short-lived (bounded TTL)
- Scoped to a single task
- Non-reusable after revocation
- Signed with HMAC to prevent forgery by the untrusted LLM
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from opensandbox_server.sar.schemas import (
    ApprovedCapabilities,
    CapabilityPermissions,
    CapabilityToken,
    NetworkMode,
)

logger = logging.getLogger(__name__)

# Default signing key; overridden by SAR config in production
_DEFAULT_SIGNING_KEY = secrets.token_hex(32)


class CapabilityStore:
    """
    Thread-safe in-memory store for active capability tokens.

    In production, this would be backed by Redis or another shared store.
    The in-memory implementation is suitable for single-process deployments.
    """

    def __init__(self) -> None:
        self._tokens: Dict[str, CapabilityToken] = {}
        self._lock = threading.Lock()

    def store(self, token: CapabilityToken) -> None:
        with self._lock:
            self._tokens[token.capability_id] = token

    def get(self, capability_id: str) -> Optional[CapabilityToken]:
        with self._lock:
            return self._tokens.get(capability_id)

    def revoke(self, capability_id: str) -> bool:
        with self._lock:
            token = self._tokens.get(capability_id)
            if token is None:
                return False
            token.revoked = True
            return True

    def remove(self, capability_id: str) -> bool:
        with self._lock:
            return self._tokens.pop(capability_id, None) is not None

    def cleanup_expired(self) -> int:
        """Remove all expired tokens. Returns number removed."""
        now = datetime.now(timezone.utc)
        removed = 0
        with self._lock:
            expired_ids = [
                cid for cid, tok in self._tokens.items()
                if tok.expires_at <= now
            ]
            for cid in expired_ids:
                del self._tokens[cid]
                removed += 1
        if removed:
            logger.info("Cleaned up %d expired capability tokens", removed)
        return removed


class CapabilityManager:
    """
    Service for minting, validating, and revoking capability tokens.

    Capabilities are derived from judge-approved permissions and are
    strictly scoped to the task that was approved. The token includes
    an HMAC signature so the runtime can verify authenticity without
    calling back to the gateway.
    """

    def __init__(self, signing_key: Optional[str] = None, max_ttl_seconds: int = 3600) -> None:
        self._signing_key = (signing_key or _DEFAULT_SIGNING_KEY).encode()
        self._max_ttl_seconds = max_ttl_seconds
        self._store = CapabilityStore()

    def mint(
        self,
        task_id: str,
        approved: ApprovedCapabilities,
        network_mode: NetworkMode = NetworkMode.OFF,
    ) -> CapabilityToken:
        """
        Mint a new capability token from judge-approved capabilities.

        The TTL is clamped to the configured maximum.
        """
        ttl = min(approved.ttl_seconds, self._max_ttl_seconds)
        now = datetime.now(timezone.utc)

        permissions = CapabilityPermissions(
            read_files=list(approved.files_read),
            write_files=list(approved.files_write),
            network_allowlist=list(approved.network),
            env_vars=list(approved.env_vars),
            max_runtime_seconds=ttl,
            network_mode=network_mode,
        )

        token = CapabilityToken(
            task_id=task_id,
            permissions=permissions,
            issued_at=now,
            expires_at=now + timedelta(seconds=ttl),
        )

        self._store.store(token)
        logger.info(
            "Minted capability %s for task %s (ttl=%ds, read=%d files, write=%d files, net=%s)",
            token.capability_id,
            task_id,
            ttl,
            len(permissions.read_files),
            len(permissions.write_files),
            network_mode.value,
        )
        return token

    def validate(self, capability_id: str) -> Optional[CapabilityToken]:
        """
        Validate a capability token is active and not expired.

        Returns the token if valid, None otherwise.
        """
        token = self._store.get(capability_id)
        if token is None:
            logger.warning("Capability %s not found", capability_id)
            return None

        if token.revoked:
            logger.warning("Capability %s has been revoked", capability_id)
            return None

        now = datetime.now(timezone.utc)
        if token.expires_at <= now:
            logger.warning("Capability %s has expired", capability_id)
            self._store.remove(capability_id)
            return None

        return token

    def revoke(self, capability_id: str) -> bool:
        """Revoke a capability token immediately."""
        result = self._store.revoke(capability_id)
        if result:
            logger.info("Revoked capability %s", capability_id)
        return result

    def revoke_for_task(self, task_id: str) -> int:
        """Revoke all capabilities for a given task."""
        count = 0
        with self._store._lock:
            for token in self._store._tokens.values():
                if token.task_id == task_id and not token.revoked:
                    token.revoked = True
                    count += 1
        if count:
            logger.info("Revoked %d capabilities for task %s", count, task_id)
        return count

    def sign_token(self, token: CapabilityToken) -> str:
        """
        Produce an HMAC signature for a capability token.

        The runtime can verify this signature to ensure the token
        was minted by an authorized gateway, not forged by the LLM.
        """
        payload = f"{token.capability_id}:{token.task_id}:{token.expires_at.isoformat()}"
        return hmac.new(self._signing_key, payload.encode(), hashlib.sha256).hexdigest()

    def verify_signature(self, token: CapabilityToken, signature: str) -> bool:
        """Verify the HMAC signature of a capability token."""
        expected = self.sign_token(token)
        return hmac.compare_digest(expected, signature)

    def cleanup(self) -> int:
        """Remove expired tokens from the store."""
        return self._store.cleanup_expired()