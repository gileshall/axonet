"""Provider registry for cloud backends."""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from .base import CloudProvider

_PROVIDERS: Dict[str, Type[CloudProvider]] = {}


def register_provider(name: str, provider_class: Type[CloudProvider]) -> None:
    """Register a cloud provider."""
    _PROVIDERS[name] = provider_class


def get_provider(name: str, **kwargs) -> CloudProvider:
    """Get and configure a cloud provider by name."""
    if name not in _PROVIDERS:
        _load_builtin_providers()
    
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider: {name}. Available: {list(_PROVIDERS.keys())}"
        )
    
    provider = _PROVIDERS[name]()
    provider.configure(**kwargs)
    return provider


def list_providers() -> List[str]:
    """List available provider names."""
    _load_builtin_providers()
    return list(_PROVIDERS.keys())


def _load_builtin_providers() -> None:
    """Lazy-load built-in providers."""
    if "local" not in _PROVIDERS:
        from .local import LocalProvider
        register_provider("local", LocalProvider)
    
    if "google" not in _PROVIDERS:
        from .google import GoogleCloudProvider
        register_provider("google", GoogleCloudProvider)
