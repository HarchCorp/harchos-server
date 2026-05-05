"""Pricing and billing API endpoints.

Provides pricing plan management, cost estimation, and billing record
access.  Pricing plans are public; billing records require authentication.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.models.pricing import Pricing, BillingRecord
from app.models.hub import Hub
from app.api.deps import require_auth
from app.schemas.common import PaginatedResponse

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas for pricing
# ---------------------------------------------------------------------------

class PricingResponse(BaseModel):
    """Pricing plan response."""

    id: str
    name: str
    gpu_type: str
    price_per_gpu_hour: float
    price_per_cpu_core_hour: float
    price_per_gb_storage_month: float
    price_per_gb_memory_hour: float
    currency: str
    region: str
    tier: str
    is_default: bool
    created_at: datetime
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class CostEstimateRequest(BaseModel):
    """Request body for cost estimation (also works as query params)."""

    gpu_count: int = Field(..., ge=1, description="Number of GPUs")
    gpu_type: str = Field("H100", description="GPU type: H100, A100, L40S")
    hours: float = Field(..., ge=0.1, description="Duration in hours")
    region: str | None = Field(None, description="Target region")
    tier: str | None = Field(None, description="Target tier")
    cpu_cores_per_gpu: int = Field(8, ge=0, description="CPU cores per GPU")
    memory_gb_per_gpu: float = Field(32.0, ge=0, description="Memory GB per GPU")
    storage_gb: float = Field(100.0, ge=0, description="Storage in GB")
    currency: str = Field("USD", description="Output currency: USD, MAD, EUR")


class CostEstimateResponse(BaseModel):
    """Cost estimation response."""

    gpu_type: str
    gpu_count: int
    hours: float
    region: str | None
    tier: str | None
    gpu_cost: float
    cpu_cost: float
    memory_cost: float
    storage_cost: float
    total_cost: float
    currency: str
    matched_plan: str | None = None


class BillingRecordResponse(BaseModel):
    """Billing record response."""

    id: str
    user_id: str
    workload_id: str | None
    hub_id: str | None
    gpu_hours: float
    cpu_core_hours: float
    memory_gb_hours: float
    storage_gb_months: float
    total_cost: float
    currency: str
    status: str
    period_start: datetime
    period_end: datetime
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Pricing plans
# ---------------------------------------------------------------------------

@router.get("/plans", response_model=PaginatedResponse[PricingResponse])
async def list_pricing_plans(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    gpu_type: str | None = Query(None, description="Filter by GPU type"),
    tier: str | None = Query(None, description="Filter by tier"),
    region: str | None = Query(None, description="Filter by region"),
    currency: str | None = Query(None, description="Filter by currency"),
    db: AsyncSession = Depends(get_db),
):
    """List all pricing plans with optional filters. Public endpoint."""
    query = select(Pricing)
    count_query = select(func.count(Pricing.id))

    if gpu_type:
        query = query.where(Pricing.gpu_type == gpu_type)
        count_query = count_query.where(Pricing.gpu_type == gpu_type)
    if tier:
        query = query.where(Pricing.tier == tier)
        count_query = count_query.where(Pricing.tier == tier)
    if region:
        query = query.where(Pricing.region == region)
        count_query = count_query.where(Pricing.region == region)
    if currency:
        query = query.where(Pricing.currency == currency)
        count_query = count_query.where(Pricing.currency == currency)

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    offset = (page - 1) * per_page
    query = query.order_by(Pricing.price_per_gpu_hour).offset(offset).limit(per_page)
    result = await db.execute(query)
    plans = result.scalars().all()

    return PaginatedResponse.create(
        items=[PricingResponse.model_validate(p) for p in plans],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/plans/{plan_id}", response_model=PricingResponse)
async def get_pricing_plan(
    plan_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific pricing plan by ID. Public endpoint."""
    result = await db.execute(select(Pricing).where(Pricing.id == plan_id))
    plan = result.scalar_one_or_none()
    if not plan:
        raise HTTPException(status_code=404, detail="Pricing plan not found")
    return PricingResponse.model_validate(plan)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

@router.get("/estimate", response_model=CostEstimateResponse)
async def estimate_cost(
    gpu_count: int = Query(..., ge=1, description="Number of GPUs"),
    gpu_type: str = Query("H100", description="GPU type: H100, A100, L40S"),
    hours: float = Query(..., ge=0.1, description="Duration in hours"),
    region: str | None = Query(None, description="Target region"),
    tier: str | None = Query(None, description="Target tier"),
    cpu_cores_per_gpu: int = Query(8, ge=0, description="CPU cores per GPU"),
    memory_gb_per_gpu: float = Query(32.0, ge=0, description="Memory GB per GPU"),
    storage_gb: float = Query(100.0, ge=0, description="Storage in GB"),
    currency: str = Query("USD", description="Output currency: USD, MAD, EUR"),
    db: AsyncSession = Depends(get_db),
):
    """Calculate a cost estimate for a given GPU configuration.

    Matches the best pricing plan based on GPU type, region, and tier.
    Falls back to a default rate if no plan is found.
    """
    # Try to find a matching pricing plan
    query = select(Pricing).where(
        Pricing.gpu_type == gpu_type,
        Pricing.currency == currency,
    )
    if region:
        query = query.where(Pricing.region == region)
    if tier:
        query = query.where(Pricing.tier == tier)

    # Prefer default plans, then cheapest
    query = query.order_by(Pricing.is_default.desc(), Pricing.price_per_gpu_hour)
    result = await db.execute(query.limit(1))
    plan = result.scalar_one_or_none()

    if plan:
        gpu_cost = plan.price_per_gpu_hour * gpu_count * hours
        cpu_cost = plan.price_per_cpu_core_hour * cpu_cores_per_gpu * gpu_count * hours
        memory_cost = plan.price_per_gb_memory_hour * memory_gb_per_gpu * gpu_count * hours
        storage_month = hours / 730.0  # ~730 hours per month
        storage_cost = plan.price_per_gb_storage_month * storage_gb * storage_month
        matched_plan = plan.name
    else:
        # Fallback default rates (USD)
        default_rates = {
            "H100": 2.10,
            "A100": 1.80,
            "L40S": 1.40,
        }
        rate = default_rates.get(gpu_type, 1.50)
        gpu_cost = rate * gpu_count * hours
        cpu_cost = 0.03 * cpu_cores_per_gpu * gpu_count * hours
        memory_cost = 0.005 * memory_gb_per_gpu * gpu_count * hours
        storage_month = hours / 730.0
        storage_cost = 0.07 * storage_gb * storage_month
        matched_plan = None

    total_cost = gpu_cost + cpu_cost + memory_cost + storage_cost

    return CostEstimateResponse(
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        hours=hours,
        region=region,
        tier=tier,
        gpu_cost=round(gpu_cost, 4),
        cpu_cost=round(cpu_cost, 4),
        memory_cost=round(memory_cost, 4),
        storage_cost=round(storage_cost, 4),
        total_cost=round(total_cost, 4),
        currency=currency,
        matched_plan=matched_plan,
    )


# ---------------------------------------------------------------------------
# Billing records
# ---------------------------------------------------------------------------

@router.get("/billing/records", response_model=PaginatedResponse[BillingRecordResponse])
async def list_billing_records(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    status: str | None = Query(None, description="Filter by status: pending, paid, overdue"),
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """List billing records for the authenticated user."""
    query = select(BillingRecord).where(BillingRecord.user_id == api_key.user_id)
    count_query = select(func.count(BillingRecord.id)).where(
        BillingRecord.user_id == api_key.user_id
    )

    if status:
        query = query.where(BillingRecord.status == status)
        count_query = count_query.where(BillingRecord.status == status)

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    offset = (page - 1) * per_page
    query = query.order_by(BillingRecord.created_at.desc()).offset(offset).limit(per_page)
    result = await db.execute(query)
    records = result.scalars().all()

    return PaginatedResponse.create(
        items=[BillingRecordResponse.model_validate(r) for r in records],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/billing/records/{record_id}", response_model=BillingRecordResponse)
async def get_billing_record(
    record_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific billing record by ID. Auth required."""
    result = await db.execute(
        select(BillingRecord).where(
            BillingRecord.id == record_id,
            BillingRecord.user_id == api_key.user_id,
        )
    )
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="Billing record not found")
    return BillingRecordResponse.model_validate(record)
