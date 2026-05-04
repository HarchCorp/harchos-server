"""Common schemas for pagination and shared types."""

from typing import Any, Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationMeta(BaseModel):
    """Pagination metadata matching the SDK's expected format."""
    total: int
    page: int
    per_page: int
    total_pages: int
    has_next: bool
    has_prev: bool


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper matching the SDK's expected format."""
    items: list[T]
    pagination: PaginationMeta

    @staticmethod
    def create(
        items: list[T],
        total: int,
        page: int,
        per_page: int,
    ) -> "PaginatedResponse[T]":
        """Helper to build a paginated response with computed metadata."""
        total_pages = max(1, (total + per_page - 1) // per_page) if total > 0 else 0
        return PaginatedResponse(
            items=items,
            pagination=PaginationMeta(
                total=total,
                page=page,
                per_page=per_page,
                total_pages=total_pages,
                has_next=page < total_pages,
                has_prev=page > 1,
            ),
        )
