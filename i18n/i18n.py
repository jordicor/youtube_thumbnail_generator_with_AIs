"""
Internationalization (i18n) System for YouTube Thumbnail Generator
===================================================================
Server-side translation system with JSON-based locale files.

Usage:
    from i18n.i18n import translate as t, set_language, get_language

    # Translate
    t('common.cancel')  # "Cancelar" or "Cancel"
    t('video.count', count=5)  # "5 videos"
"""

import json
from pathlib import Path
from typing import Any, Optional
from functools import lru_cache
from contextvars import ContextVar

# Project root
PROJECT_DIR = Path(__file__).parent.parent

# Locales directory (shared with frontend in static/locales/)
LOCALES_DIR = PROJECT_DIR / "static" / "locales"

# Supported languages
SUPPORTED_LANGUAGES = ["es", "en"]
DEFAULT_LANGUAGE = "es"

# Current language (using ContextVar for async-safe per-request isolation)
_current_language: ContextVar[str] = ContextVar('current_language', default=DEFAULT_LANGUAGE)

# Cache for loaded translations
_translations_cache: dict[str, dict] = {}


def _load_translations(lang: str) -> dict:
    """Load translations from JSON file for a language."""
    if lang in _translations_cache:
        return _translations_cache[lang]

    locale_file = LOCALES_DIR / f"{lang}.json"

    if not locale_file.exists():
        print(f"Warning: Locale file not found: {locale_file}")
        return {}

    try:
        with open(locale_file, "r", encoding="utf-8") as f:
            translations = json.load(f)
            _translations_cache[lang] = translations
            return translations
    except json.JSONDecodeError as e:
        print(f"Error parsing locale file {locale_file}: {e}")
        return {}


def _get_nested_value(data: dict, key_path: str) -> Optional[str]:
    """Get a nested value from a dictionary using dot notation.

    Example: _get_nested_value(data, 'common.buttons.cancel')
    """
    keys = key_path.split(".")
    current = data

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None

    return current if isinstance(current, str) else None


def _format_string(template: str, **kwargs) -> str:
    """Format a string with placeholders.

    Supports {name} style placeholders.
    """
    if not kwargs:
        return template

    try:
        return template.format(**kwargs)
    except KeyError:
        # Return template as-is if placeholder is missing
        return template


def _get_plural_key(key: str, count: int, lang: str) -> str:
    """Get the correct plural key based on count and language.

    Follows CLDR plural rules (simplified):
    - English: one (1), other (everything else)
    - Spanish: one (1), other (everything else)

    Keys should be named: key_one, key_other
    """
    if count == 1:
        return f"{key}_one"
    else:
        return f"{key}_other"


def get_language() -> str:
    """Get current language."""
    return _current_language.get()


def set_language(lang: str) -> bool:
    """Set current language.

    Returns True if language was set, False if not supported.
    """
    if lang in SUPPORTED_LANGUAGES:
        _current_language.set(lang)
        return True

    return False


def translate(key: str, count: Optional[int] = None, **kwargs) -> str:
    """Translate a key to the current language.

    Args:
        key: Dot-notation key (e.g., 'common.cancel')
        count: If provided, uses plural rules
        **kwargs: Placeholder values

    Returns:
        Translated string, or key if not found
    """
    lang = _current_language.get()
    translations = _load_translations(lang)

    # Handle pluralization
    if count is not None:
        plural_key = _get_plural_key(key, count, lang)
        value = _get_nested_value(translations, plural_key)

        # Fallback to base key if plural not found
        if value is None:
            value = _get_nested_value(translations, key)

        # Add count to kwargs for formatting
        kwargs["count"] = count
    else:
        value = _get_nested_value(translations, key)

    # Fallback to English if not found in current language
    if value is None and lang != "en":
        en_translations = _load_translations("en")
        if count is not None:
            plural_key = _get_plural_key(key, count, "en")
            value = _get_nested_value(en_translations, plural_key)
            if value is None:
                value = _get_nested_value(en_translations, key)
        else:
            value = _get_nested_value(en_translations, key)

    # If still not found, return the key itself (useful for debugging)
    if value is None:
        return f"[{key}]"

    return _format_string(value, **kwargs)


def get_translator():
    """Get the translate function.

    This is the main function to use in templates and code.
    """
    return translate


def reload_translations():
    """Clear translation cache and reload from files.

    Useful during development.
    """
    global _translations_cache
    _translations_cache = {}


def get_all_translations(lang: Optional[str] = None) -> dict:
    """Get all translations for a language.

    Useful for sending to frontend.
    """
    if lang is None:
        lang = _current_language.get()

    return _load_translations(lang)


# Shorthand alias
t = translate
