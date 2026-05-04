"""Energy Pydantic schemas matching the SDK model."""

from datetime import datetime

from pydantic import BaseModel

class EnergyReportResponse(BaseModel):
    """Energy report response."""
    id: str
    resource_id: str
    resource_type: str
    total_consumption_kwh: float
    renewable_percentage: float
    carbon_emissions_kg: float
    pue: float
    efficiency_score: float
    period_start: datetime
    period_end: datetime
    created_at: datetime

    model_config = {'from_attributes': True}

class EnergySummaryResponse(BaseModel):
    """Energy summary – matches SDK EnergySummary model.

    Required fields must align with the SDK's ``EnergySummary``:
    total_kwh, total_co2_grams, average_pue, average_renewable_fraction,
    resource_count, period_start, period_end.
    """

    total_kwh: float
    total_co2_grams: float
    average_pue: float
    average_renewable_fraction: float
    resource_count: int
    period_start: datetime
    period_end: datetime

class GreenWindowResponse(BaseModel):
    """A green energy window for scheduling — matches SDK GreenWindow model."""
    hub_id: str
    hub_name: str
    start: datetime
    end: datetime
    renewable_percentage: float
    estimated_co2_grams_per_kwh: float
    recommended: bool

class EnergyConsumptionResponse(BaseModel):
    """Energy consumption data point."""
    id: str
    resource_id: str
    resource_type: str
    consumption_kwh: float
    carbon_emissions_kg: float
    renewable_percentage: float
    timestamp: datetime

    model_config = {'from_attributes': True}
