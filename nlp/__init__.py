"""Shared natural-language processing primitives.

The package intentionally keeps its core text contracts dependency-free.  Model
adapters and optional third-party integrations live in their own modules.
"""

from .text import (
    ProtectedSpan,
    TextPreparationError,
    TokenSpan,
    normalize_processing_text,
    protection_spans,
    script_bucket,
    tokenize_with_spans,
)

__all__ = [
    "ProtectedSpan",
    "TextPreparationError",
    "TokenSpan",
    "normalize_processing_text",
    "protection_spans",
    "script_bucket",
    "tokenize_with_spans",
]
