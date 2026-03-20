"""Multi-provider data abstraction layer."""

from providers.registry import get_provider_chain

__all__ = ["get_provider_chain"]
