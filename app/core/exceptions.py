"""Structured error handling with HarchOS error codes.

Inspired by Replicate's E#### error code system — every error has a
machine-readable code that clients can match on, plus a human-readable
message and a doc URL for remediation.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("harchos.errors")

# ---------------------------------------------------------------------------
# Error code registry — E0xxx = client errors, E1xxx = server errors
# ---------------------------------------------------------------------------

ERROR_CODES: dict[str, dict[str, Any]] = {
    # Auth errors (E01xx)
    "E0100": {"status": 401, "title": "Authentication Required", "doc": "/docs/errors/E0100"},
    "E0101": {"status": 401, "title": "Invalid API Key", "doc": "/docs/errors/E0101"},
    "E0102": {"status": 401, "title": "Invalid or Expired Token", "doc": "/docs/errors/E0102"},
    "E0103": {"status": 401, "title": "Email Does Not Match API Key Owner", "doc": "/docs/errors/E0103"},
    "E0104": {"status": 403, "title": "Insufficient Permissions", "doc": "/docs/errors/E0104"},
    "E0105": {"status": 403, "title": "Registration Disabled in Production", "doc": "/docs/errors/E0105"},
    "E0106": {"status": 403, "title": "Resource Access Denied", "doc": "/docs/errors/E0106"},

    # Validation errors (E02xx)
    "E0200": {"status": 400, "title": "Validation Error", "doc": "/docs/errors/E0200"},
    "E0201": {"status": 400, "title": "Invalid Enum Value", "doc": "/docs/errors/E0201"},
    "E0202": {"status": 400, "title": "Model Not Found", "doc": "/docs/errors/E0202"},
    "E0203": {"status": 400, "title": "Hub Not Ready", "doc": "/docs/errors/E0203"},
    "E0204": {"status": 422, "title": "Unprocessable Entity", "doc": "/docs/errors/E0204"},

    # Resource errors (E03xx)
    "E0300": {"status": 404, "title": "Resource Not Found", "doc": "/docs/errors/E0300"},
    "E0301": {"status": 404, "title": "Workload Not Found", "doc": "/docs/errors/E0301"},
    "E0302": {"status": 404, "title": "Hub Not Found", "doc": "/docs/errors/E0302"},
    "E0303": {"status": 404, "title": "Model Not Found", "doc": "/docs/errors/E0303"},
    "E0304": {"status": 404, "title": "User Not Found", "doc": "/docs/errors/E0304"},
    "E0305": {"status": 404, "title": "API Key Not Found", "doc": "/docs/errors/E0305"},
    "E0306": {"status": 404, "title": "Billing Record Not Found", "doc": "/docs/errors/E0306"},
    "E0307": {"status": 404, "title": "Carbon Data Not Available", "doc": "/docs/errors/E0307"},
    "E0308": {"status": 409, "title": "Resource Already Exists", "doc": "/docs/errors/E0308"},
    "E0309": {"status": 409, "title": "Email Already Registered", "doc": "/docs/errors/E0309"},

    # Rate limiting (E04xx)
    "E0400": {"status": 429, "title": "Rate Limit Exceeded", "doc": "/docs/errors/E0400"},
    "E0401": {"status": 429, "title": "Concurrent Request Limit Exceeded", "doc": "/docs/errors/E0401"},
    "E0402": {"status": 429, "title": "Carbon Budget Exceeded", "doc": "/docs/errors/E0402"},

    # Inference errors (E05xx)
    "E0500": {"status": 503, "title": "Inference Service Unavailable", "doc": "/docs/errors/E0500"},
    "E0501": {"status": 504, "title": "Inference Timeout", "doc": "/docs/errors/E0501"},
    "E0502": {"status": 400, "title": "Inference Model Not Available", "doc": "/docs/errors/E0502"},
    "E0503": {"status": 429, "title": "Inference Capacity Exceeded", "doc": "/docs/errors/E0503"},
    "E0504": {"status": 400, "title": "Context Length Exceeded", "doc": "/docs/errors/E0504"},

    # Server errors (E1xxx)
    "E1000": {"status": 500, "title": "Internal Server Error", "doc": "/docs/errors/E1000"},
    "E1001": {"status": 502, "title": "Upstream Service Error", "doc": "/docs/errors/E1001"},
    "E1002": {"status": 503, "title": "Service Unavailable", "doc": "/docs/errors/E1002"},
    "E1003": {"status": 503, "title": "Database Connection Error", "doc": "/docs/errors/E1003"},
    "E1004": {"status": 503, "title": "Cache Service Error", "doc": "/docs/errors/E1004"},
}


class ErrorDetail(BaseModel):
    """Structured error detail following RFC 7807-inspired format."""
    code: str
    title: str
    detail: str
    doc_url: str | None = None
    meta: dict[str, Any] | None = None


class HarchOSError(HTTPException):
    """Base exception for all HarchOS errors with structured error codes.

    Usage:
        raise HarchOSError("E0301", detail="Workload abc-123 not found")
        raise HarchOSError("E0104", detail="Only admins can create hubs", meta={"required_role": "admin"})
    """

    def __init__(
        self,
        code: str,
        detail: str = "",
        meta: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ):
        if code not in ERROR_CODES:
            logger.warning("Unknown error code: %s, falling back to E1000", code)
            code = "E1000"

        entry = ERROR_CODES[code]
        self.error_code = code
        self.error_title = entry["title"]
        self.error_detail = detail or entry["title"]
        self.error_doc = entry.get("doc")
        self.error_meta = meta

        super().__init__(
            status_code=entry["status"],
            detail=self.error_detail,
            headers=headers,
        )

    def to_response(self) -> dict[str, Any]:
        """Convert to JSON response body."""
        result = {
            "error": {
                "code": self.error_code,
                "title": self.error_title,
                "detail": self.error_detail,
            }
        }
        if self.error_doc:
            result["error"]["doc_url"] = self.error_doc
        if self.error_meta:
            result["error"]["meta"] = self.error_meta
        return result


# ---------------------------------------------------------------------------
# Convenience constructors for common errors
# ---------------------------------------------------------------------------

def auth_required() -> HarchOSError:
    return HarchOSError("E0100", detail="Authentication required. Provide X-API-Key header or Bearer token.")


def invalid_api_key() -> HarchOSError:
    return HarchOSError("E0101", detail="The provided API key is invalid or has been revoked.")


def invalid_token() -> HarchOSError:
    return HarchOSError("E0102", detail="The provided token is invalid or has expired.")


def insufficient_permissions(required_role: str = "admin") -> HarchOSError:
    return HarchOSError("E0104", detail=f"Insufficient permissions. Required role: {required_role}", meta={"required_role": required_role})


def resource_access_denied(resource_type: str, resource_id: str) -> HarchOSError:
    return HarchOSError("E0106", detail=f"You do not have access to {resource_type} '{resource_id}'.", meta={"resource_type": resource_type})


def not_found(resource_type: str, resource_id: str = "") -> HarchOSError:
    code_map = {
        "workload": "E0301", "hub": "E0302", "model": "E0303",
        "user": "E0304", "api_key": "E0305", "billing": "E0306",
    }
    code = code_map.get(resource_type, "E0300")
    msg = f"{resource_type.capitalize()} not found" + (f": {resource_id}" if resource_id else "")
    return HarchOSError(code, detail=msg, meta={"resource_type": resource_type, "resource_id": resource_id})


def rate_limit_exceeded(retry_after: int = 60) -> HarchOSError:
    return HarchOSError("E0400", detail="Rate limit exceeded. Please slow down.", meta={"retry_after_seconds": retry_after})


def validation_error(field: str, reason: str) -> HarchOSError:
    return HarchOSError("E0200", detail=f"Validation error on '{field}': {reason}", meta={"field": field})


def invalid_enum_value(field: str, value: str, allowed: list[str]) -> HarchOSError:
    return HarchOSError("E0201", detail=f"Invalid value '{value}' for '{field}'. Allowed: {', '.join(allowed)}", meta={"field": field, "allowed": allowed})


def model_not_available(model_id: str) -> HarchOSError:
    return HarchOSError("E0502", detail=f"Model '{model_id}' is not available for inference.")


def inference_timeout(model_id: str, timeout_seconds: int = 30) -> HarchOSError:
    return HarchOSError("E0501", detail=f"Inference request to model '{model_id}' timed out after {timeout_seconds}s.")


def carbon_budget_exceeded(budget_grams: float, actual_grams: float) -> HarchOSError:
    return HarchOSError("E0402", detail=f"Carbon budget exceeded: {actual_grams:.2f}g used of {budget_grams:.2f}g budget.", meta={"budget_grams": budget_grams, "actual_grams": actual_grams})


def already_exists(resource_type: str, identifier: str) -> HarchOSError:
    return HarchOSError("E0308", detail=f"A {resource_type} with this {identifier} already exists.")


# ---------------------------------------------------------------------------
# Global exception handler — catches all HarchOSError and formats response
# ---------------------------------------------------------------------------

async def harchos_error_handler(request: Request, exc: HarchOSError) -> JSONResponse:
    """FastAPI exception handler for HarchOSError."""
    response = exc.to_response()

    headers = {}
    if exc.error_code == "E0400" and exc.error_meta and "retry_after_seconds" in exc.error_meta:
        headers["Retry-After"] = str(exc.error_meta["retry_after_seconds"])

    return JSONResponse(
        status_code=exc.status_code,
        content=response,
        headers=headers,
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions — returns E1000 without leaking internals."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "E1000",
                "title": "Internal Server Error",
                "detail": "An unexpected error occurred. Please try again or contact support@harchos.ai.",
            }
        },
    )
