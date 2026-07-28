"""Process-wide switch for KV-cache reuse during autoregressive sampling.

Sampling walks the transformer one token at a time while re-feeding the whole
prefix, so the attention keys/values of the prefix are recomputed at every step.
Enabling this switch makes the generators keep the transformer cache between
steps and feed only the newly available token.

The switch is global rather than a `sample()` argument because the nested
per-level sample callables are bound with `functools.partial` and cannot carry
extra keyword arguments.
"""

from contextlib import contextmanager

_enabled = False


def is_enabled() -> bool:
    """Whether sampling should reuse transformer KV caches."""
    return _enabled


# Kept as a distinct name so call sites reading it for cache bookkeeping stay
# readable alongside `is_enabled`, which also gates inference-only shortcuts.
use_kv_cache = is_enabled


@contextmanager
def kv_cache(enabled: bool = True):
    """Enable (or explicitly disable) KV-cache reuse for the enclosed block."""
    global _enabled
    previous = _enabled
    _enabled = enabled
    try:
        yield
    finally:
        _enabled = previous
