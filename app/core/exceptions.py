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
    "E0310": {"status": 404, "title": "Project Not Found", "doc": "/docs/errors/E0310"},
    "E0311": {"status": 403, "title": "Project Access Denied", "doc": "/docs/errors/E0311"},
    "E0312": {"status": 403, "title": "Insufficient Scope", "doc": "/docs/errors/E0312"},
    "E0313": {"status": 429, "title": "Token Budget Exceeded", "doc": "/docs/errors/E0313"},
    "E0314": {"status": 429, "title": "Spending Limit Exceeded", "doc": "/docs/errors/E0314"},
    "E0315": {"status": 400, "title": "Model Not Allowed For Key", "doc": "/docs/errors/E0315"},
    "E0316": {"status": 400, "title": "Region Not Allowed For Key", "doc": "/docs/errors/E0316"},
    "E0317": {"status": 403, "title": "Project Inactive", "doc": "/docs/errors/E0317"},

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
    "E0505": {"status": 400, "title": "Invalid Embedding Input", "doc": "/docs/errors/E0505"},
    "E0506": {"status": 400, "title": "Batch Size Exceeded", "doc": "/docs/errors/E0506"},
    "E0507": {"status": 400, "title": "Batch Item Error", "doc": "/docs/errors/E0507"},

    # Fine-tuning errors (E06xx)
    "E0600": {"status": 400, "title": "Fine-Tuning Job Error", "doc": "/docs/errors/E0600"},
    "E0601": {"status": 400, "title": "Invalid Training Data Format", "doc": "/docs/errors/E0601"},
    "E0602": {"status": 400, "title": "Training File Too Large", "doc": "/docs/errors/E0602"},
    "E0603": {"status": 404, "title": "Fine-Tuning Job Not Found", "doc": "/docs/errors/E0603"},
    "E0604": {"status": 400, "title": "Fine-Tuning Job Cannot Be Cancelled", "doc": "/docs/errors/E0604"},
    "E0605": {"status": 400, "title": "Carbon Budget Exceeded During Training", "doc": "/docs/errors/E0605"},
    "E0606": {"status": 404, "title": "Training File Not Found", "doc": "/docs/errors/E0606"},
    "E0607": {"status": 400, "title": "Unsupported Base Model For Fine-Tuning", "doc": "/docs/errors/E0607"},

    # WebSocket errors (E07xx)
    "E0700": {"status": 400, "title": "WebSocket Error", "doc": "/docs/errors/E0700"},
    "E0701": {"status": 401, "title": "WebSocket Authentication Required", "doc": "/docs/errors/E0701"},
    "E0702": {"status": 429, "title": "WebSocket Connection Limit Exceeded", "doc": "/docs/errors/E0702"},

    # Health subsystem errors (E08xx)
    "E0800": {"status": 503, "title": "Service Not Ready", "doc": "/docs/errors/E0800"},
    "E0801": {"status": 503, "title": "Startup Incomplete", "doc": "/docs/errors/E0801"},
    "E0802": {"status": 503, "title": "Database Unhealthy", "doc": "/docs/errors/E0802"},
    "E0803": {"status": 503, "title": "Cache Unhealthy", "doc": "/docs/errors/E0803"},
    "E0804": {"status": 503, "title": "Inference Backend Unhealthy", "doc": "/docs/errors/E0804"},
    "E0805": {"status": 503, "title": "Carbon API Unhealthy", "doc": "/docs/errors/E0805"},

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
        "project": "E0310",
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


def project_not_found(project_id: str) -> HarchOSError:
    return HarchOSError("E0310", detail=f"Project not found: {project_id}", meta={"project_id": project_id})


def project_access_denied(project_id: str) -> HarchOSError:
    return HarchOSError("E0311", detail=f"You do not have access to project '{project_id}'.", meta={"project_id": project_id})


def insufficient_scope(required_scope: str) -> HarchOSError:
    return HarchOSError("E0312", detail=f"API key lacks required scope: {required_scope}", meta={"required_scope": required_scope})


def token_budget_exceeded(used: int, limit: int) -> HarchOSError:
    return HarchOSError("E0313", detail=f"Daily token budget exceeded: {used} used of {limit} limit.", meta={"tokens_used": used, "tokens_limit": limit})


def spending_limit_exceeded(spent: float, limit: float) -> HarchOSError:
    return HarchOSError("E0314", detail=f"Monthly spending limit exceeded: ${spent:.2f} spent of ${limit:.2f} limit.", meta={"spent_usd": spent, "limit_usd": limit})


def model_not_allowed(model_id: str) -> HarchOSError:
    return HarchOSError("E0315", detail=f"Model '{model_id}' is not allowed for this API key.", meta={"model_id": model_id})


def region_not_allowed(region: str) -> HarchOSError:
    return HarchOSError("E0316", detail=f"Region '{region}' is not allowed for this API key.", meta={"region": region})


def project_inactive(project_id: str) -> HarchOSError:
    return HarchOSError("E0317", detail=f"Project '{project_id}' is inactive. Reactivate it to use its API keys.", meta={"project_id": project_id})


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
