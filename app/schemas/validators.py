"""Comprehensive input validators for HarchOS API.

These validators enforce strict input sanitization across all endpoints,
preventing injection attacks, ensuring data integrity, and providing
clear error messages. Every field that touches the database or external
services is validated here.

Inspired by:
- Stripe API: strict input validation with clear error codes
- Replicate: per-model input schema validation
- Together AI: OpenAI-compatible request validation with extensions
"""

import re
import html
from typing import Any

from pydantic import field_validator, model_validator


# ---------------------------------------------------------------------------
# Common field validators (reusable across schemas)
# ---------------------------------------------------------------------------

def validate_name(v: str) -> str:
    """Validate a resource name: 1-128 chars, no control chars, trimmed."""
    if not v or not v.strip():
        raise ValueError("Name must not be empty or whitespace-only")
    v = v.strip()
    if len(v) > 128:
        raise ValueError("Name must be at most 128 characters")
    # Reject control characters
    if re.search(r'[\x00-\x1f\x7f]', v):
        raise ValueError("Name must not contain control characters")
    return v


def validate_email_field(v: str) -> str:
    """Validate and normalize an email address."""
    v = v.strip().lower()
    # RFC 5322 simplified pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, v):
        raise ValueError(f"Invalid email format: {v}")
    if len(v) > 254:
        raise ValueError("Email must be at most 254 characters")
    return v


def validate_string_field(v: str, field_name: str = "field", max_len: int = 1000) -> str:
    """Validate a generic string field: sanitize HTML, trim, length check."""
    if not v:
        return v
    v = v.strip()
    if len(v) > max_len:
        raise ValueError(f"{field_name} must be at most {max_len} characters")
    # Sanitize HTML to prevent XSS
    sanitized = html.escape(v)
    if sanitized != v:
        raise ValueError(f"{field_name} must not contain HTML or special characters")
    return v


def validate_labels(v: dict[str, str]) -> dict[str, str]:
    """Validate Kubernetes-style labels.

    Keys: prefix (optional) + name, following DNS subdomain name rules.
    Values: must be <= 63 chars.
    Max 50 labels per resource.
    """
    if not v:
        return v
    if len(v) > 50:
        raise ValueError("Maximum 50 labels allowed per resource")

    label_key_pattern = re.compile(
        r'^([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)*[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(/[A-Za-z0-9]([A-Za-z0-9._-]{0,61}[A-Za-z0-9])?)?$'
    )

    for key, value in v.items():
        if len(key) > 63:
            raise ValueError(f"Label key too long (max 63 chars): {key[:30]}...")
        if not label_key_pattern.match(key) and not re.match(r'^[a-zA-Z0-9_.-]+$', key):
            raise ValueError(f"Invalid label key: {key}")
        if len(value) > 63:
            raise ValueError(f"Label value too long (max 63 chars) for key '{key}'")
        if re.search(r'[\x00-\x1f\x7f]', value):
            raise ValueError(f"Label value for key '{key}' must not contain control characters")
    return v


def validate_annotations(v: dict[str, str]) -> dict[str, str]:
    """Validate Kubernetes-style annotations (more permissive than labels)."""
    if not v:
        return v
    if len(v) > 50:
        raise ValueError("Maximum 50 annotations allowed per resource")

    for key, value in v.items():
        if len(key) > 253:
            raise ValueError(f"Annotation key too long (max 253 chars): {key[:30]}...")
        if len(value) > 1048576:  # 1MB
            raise ValueError(f"Annotation value too long for key '{key}'")
    return v


def validate_gpu_type(v: str) -> str:
    """Validate GPU type against supported hardware."""
    supported = {
        "a100", "a100-80gb", "h100", "h100-80gb", "h200", "b200",
        "l40s", "l4", "a10", "a10g", "t4", "v100",
        "rtx4090", "rtx3090", "rtx6000-ada", "a6000",
    }
    normalized = v.strip().lower()
    if normalized not in supported:
        raise ValueError(
            f"Unsupported GPU type: {v}. Supported: {', '.join(sorted(supported))}"
        )
    return normalized


def validate_region(v: str) -> str:
    """Validate region identifier."""
    v = v.strip().lower()
    valid_regions = {
        "morocco", "africa-north", "africa-west", "africa-east", "africa-south",
        "europe-west", "europe-north", "middle-east", "us-east", "us-west",
        "asia-east", "asia-southeast",
    }
    if v not in valid_regions:
        raise ValueError(
            f"Invalid region: {v}. Supported: {', '.join(sorted(valid_regions))}"
        )
    return v


def validate_command_list(v: list[str] | None) -> list[str] | None:
    """Validate a command list: each element is a non-empty string, max 100 elements."""
    if v is None:
        return v
    if len(v) > 100:
        raise ValueError("Command list must have at most 100 elements")
    for i, arg in enumerate(v):
        if not isinstance(arg, str):
            raise ValueError(f"Command argument {i} must be a string")
        if len(arg) > 4096:
            raise ValueError(f"Command argument {i} too long (max 4096 chars)")
        # Prevent shell injection in commands
        if re.search(r'[\x00\x0a\x0d]', arg):
            raise ValueError(f"Command argument {i} must not contain newline or null characters")
    return v


def validate_env_dict(v: dict[str, str]) -> dict[str, str]:
    """Validate environment variable dict: valid keys, safe values."""
    if not v:
        return v
    if len(v) > 100:
        raise ValueError("Maximum 100 environment variables allowed")

    env_key_pattern = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

    for key, value in v.items():
        if not env_key_pattern.match(key):
            raise ValueError(f"Invalid environment variable name: {key}")
        if len(key) > 128:
            raise ValueError(f"Environment variable name too long: {key[:30]}...")
        if len(value) > 8192:
            raise ValueError(f"Environment variable value too long for key '{key}'")
        # Block secrets in plain env vars (heuristic)
        secret_patterns = ['password', 'secret', 'private_key', 'token']
        if any(p in key.lower() for p in secret_patterns):
            raise ValueError(
                f"Environment variable '{key}' appears to contain sensitive data. "
                f"Use the secrets management endpoint instead."
            )
    return v


def validate_url(v: str | None) -> str | None:
    """Validate a URL field."""
    if v is None:
        return v
    v = v.strip()
    if len(v) > 2048:
        raise ValueError("URL must be at most 2048 characters")
    url_pattern = re.compile(
        r'^https?://[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*(:[0-9]+)?(/.*)?$'
    )
    if not url_pattern.match(v):
        raise ValueError(f"Invalid URL format: {v[:50]}...")
    return v


def validate_api_key_name(v: str) -> str:
    """Validate API key name: 1-64 chars, alphanumeric + limited special chars."""
    v = v.strip()
    if not v:
        raise ValueError("API key name must not be empty")
    if len(v) > 64:
        raise ValueError("API key name must be at most 64 characters")
    if not re.match(r'^[a-zA-Z0-9._\-\s]+$', v):
        raise ValueError("API key name can only contain letters, numbers, spaces, dots, hyphens, and underscores")
    return v


def validate_positive_int(v: int, field_name: str = "value", max_val: int = 10000) -> int:
    """Validate a positive integer within bounds."""
    if v < 0:
        raise ValueError(f"{field_name} must be non-negative")
    if v > max_val:
        raise ValueError(f"{field_name} must be at most {max_val}")
    return v


def validate_positive_float(v: float, field_name: str = "value", max_val: float = 1e9) -> float:
    """Validate a positive float within bounds."""
    if v < 0:
        raise ValueError(f"{field_name} must be non-negative")
    if v > max_val:
        raise ValueError(f"{field_name} must be at most {max_val}")
    return v


def sanitize_string(v: str) -> str:
    """Sanitize a string by trimming and removing control characters."""
    if not v:
        return v
    v = v.strip()
    # Remove control characters except newline and tab
    v = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', v)
    return v
