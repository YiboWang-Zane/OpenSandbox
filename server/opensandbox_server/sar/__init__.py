"""
Secure Agentic Runtime (SAR) module for OpenSandbox

Implements a security gateway architecture where every LLM-generated tool call
is treated as an untrusted request that must pass through validation, capability
minting, and monitored ephemeral execution before any side effects occur.

Core components:
- Gateway: central execution gateway that intercepts all tool call requests
- Judge: semantic validation of tool calls against user intent
- Capabilities: temporal, task-scoped capability tokens
- Ephemeral: zero-state runtime lifecycle management
- ResourceProjection: file mount isolation and path hiding
- NetworkPolicy: per-task network access control
- Monitor: deterministic syscall/process monitoring via seccomp and eBPF
"""