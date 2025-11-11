# utils/attacks/__init__.py
"""
Attack registry and factory.

This minimal registry avoids importing optional attack modules that may not
exist in the tree. By default it returns None (baseline / no attack).
"""

from typing import Optional

# Prefer the canonical base; fall back to 'basy' if you renamed the file.
try:
    from .base import AttackBase  # noqa: F401
except Exception:
    try:
        from .basy import AttackBase  # type: ignore  # noqa: F401
    except Exception:
        class AttackBase:  # type: ignore
            def __init__(self, *args, **kwargs):
                pass

__all__ = [
    "AttackBase",
    "build_attack",
]

# Minimal registry: only entries that are actually available.
# If you add back any attack modules, import and register them here.
_REGISTRY = {
    "none": None,
    # Legacy names mapped to None for robustness (won't raise if passed in configs):
}


def build_attack(args) -> Optional[AttackBase]:
    """
    Return an attack instance or None, based on args.attack.
    Unknown or missing names gracefully fall back to None.
    """
    name = str(getattr(args, "attack", "none") or "none").lower()
    cls = _REGISTRY.get(name, None)
    return None if cls is None else cls(args)
