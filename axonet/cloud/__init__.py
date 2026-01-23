"""Cloud infrastructure abstraction for axonet.

Supports local execution and Google Cloud, with extensibility for other providers.
"""

from .base import (
    CloudProvider,
    StorageBackend,
    ComputeBackend,
    BatchBackend,
    JobConfig,
    JobStatus,
)
from .registry import get_provider, register_provider, list_providers

__all__ = [
    "CloudProvider",
    "StorageBackend",
    "ComputeBackend",
    "BatchBackend",
    "JobConfig",
    "JobStatus",
    "get_provider",
    "register_provider",
    "list_providers",
]
