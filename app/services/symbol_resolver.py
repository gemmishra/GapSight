"""Resolve user-provided symbol names to canonical symbols."""

from typing import Optional

from app.utils.constants import SYMBOL_ALIASES, SUPPORTED_SYMBOL_IDS


def normalize_symbol(symbol: str) -> str:
    """Normalize whitespace and casing in a symbol string."""
    return " ".join(symbol.strip().upper().split())


def resolve_symbol(symbol: str) -> Optional[str]:
    """Return the canonical symbol when it is currently supported."""
    canonical_symbol = SYMBOL_ALIASES.get(normalize_symbol(symbol))
    if canonical_symbol in SUPPORTED_SYMBOL_IDS:
        return canonical_symbol
    return None


def is_symbol_supported(symbol: str) -> bool:
    """Return whether a symbol resolves to an active supported symbol."""
    return resolve_symbol(symbol) is not None
