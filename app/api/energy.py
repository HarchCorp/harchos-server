"""Energy endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.schemas.energy import (
    EnergyReportResponse,
    EnergySummaryResponse,
    GreenWindowResponse,
    EnergyConsumptionResponse,
)
from app.services.energy_service import EnergyService
from app.api.deps import require_auth

router = APIRouter()

@router.get("/reports/{resource_id}", response_model=EnergyReportResponse)
async def get_energy_report(
    resource_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get energy report for a resource (hub)."""
    result = await EnergyService.get_energy_report(db, resource_id)
    if not result:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Energy report not found for resource")
    return result

@router.get("/summary", response_model=EnergySummaryResponse)
async def get_energy_summary(
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get energy summary across all hubs."""
    return await EnergyService.get_energy_summary(db)

@router.get("/green-windows", response_model=list[GreenWindowResponse])
async def get_green_windows(
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get green energy windows for scheduling."""
    return await EnergyService.get_green_windows(db)

@router.get("/consumption/{resource_id}", response_model=list[EnergyConsumptionResponse])
async def get_consumption(
    resource_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get energy consumption data for a resource."""
    return await EnergyService.get_consumption(db, resource_id)
