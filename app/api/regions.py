"""Regions API endpoint.

Returns a catalog of available deployment regions across Africa with
metadata including hub counts, GPU capacity, renewable energy percentages,
carbon intensity, latency from Morocco, and compliance frameworks.

Data is sourced from the hubs database (no more hardcoded lists).
"""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.models.hub import Hub
from app.api.deps import require_auth

router = APIRouter()


# ---------------------------------------------------------------------------
# Region data models
# ---------------------------------------------------------------------------

class RegionInfo(BaseModel):
    """Information about a deployment region."""

    name: str = Field(..., description="Full region name")
    code: str = Field(..., description="Region code (ISO 3166-1 alpha-2)")
    country: str = Field(..., description="Country name")
    available: bool = Field(True, description="Whether the region is available for deployment")
    hub_count: int = Field(0, description="Number of GPU hubs in this region")
    total_gpus: int = Field(0, description="Total GPUs across all hubs in the region")
    avg_renewable_percentage: float = Field(0.0, description="Average renewable energy percentage")
    avg_carbon_intensity: float = Field(0.0, description="Average grid carbon intensity (gCO2/kWh)")
    latency_ms_from_casablanca: float = Field(0.0, description="Average latency in ms from Casablanca")
    compliance_frameworks: list[str] = Field(
        default_factory=list,
        description="Compliance frameworks supported in this region",
    )


class RegionListResponse(BaseModel):
    """Response for the regions list endpoint."""

    regions: list[RegionInfo]
    total: int


# ---------------------------------------------------------------------------
# Static region metadata — hub data comes from DB, compliance/latency is regional
# ---------------------------------------------------------------------------

# Compliance frameworks and latency per region (not stored in DB hubs)
REGION_METADATA: dict[str, dict] = {
    "MA": {
        "name": "Morocco",
        "country": "Morocco",
        "latency_ms_from_casablanca": 0.0,
        "compliance_frameworks": [
            "ISO 27001", "GDPR", "CNDP (Moroccan DPA)", "Sovereign Cloud", "PCI DSS",
        ],
    },
    "DZ": {
        "name": "Algeria",
        "country": "Algeria",
        "latency_ms_from_casablanca": 22.0,
        "compliance_frameworks": ["ANDI", "GDPR (partial)"],
    },
    "TN": {
        "name": "Tunisia",
        "country": "Tunisia",
        "latency_ms_from_casablanca": 28.0,
        "compliance_frameworks": ["INPDP (Tunisian DPA)", "GDPR (partial)"],
    },
    "EG": {
        "name": "Egypt",
        "country": "Egypt",
        "latency_ms_from_casablanca": 55.0,
        "compliance_frameworks": ["EITDA", "PCI DSS", "ISO 27001"],
    },
    "NG": {
        "name": "Nigeria",
        "country": "Nigeria",
        "latency_ms_from_casablanca": 85.0,
        "compliance_frameworks": ["NDPR", "NITDA", "ISO 27001"],
    },
    "KE": {
        "name": "Kenya",
        "country": "Kenya",
        "latency_ms_from_casablanca": 110.0,
        "compliance_frameworks": ["DPA 2019", "ISO 27001", "PCI DSS"],
    },
    "ZA": {
        "name": "South Africa",
        "country": "South Africa",
        "latency_ms_from_casablanca": 145.0,
        "compliance_frameworks": ["POPIA", "PCI DSS", "ISO 27001", "SOC 2"],
    },
    "SN": {
        "name": "Senegal",
        "country": "Senegal",
        "latency_ms_from_casablanca": 35.0,
        "compliance_frameworks": ["ADP (Senegalese DPA)", "ISO 27001"],
    },
    "GH": {
        "name": "Ghana",
        "country": "Ghana",
        "latency_ms_from_casablanca": 50.0,
        "compliance_frameworks": ["DPA 2012", "ISO 27001"],
    },
    "ET": {
        "name": "Ethiopia",
        "country": "Ethiopia",
        "latency_ms_from_casablanca": 95.0,
        "compliance_frameworks": ["FDRE Proclamation"],
    },
    "CI": {
        "name": "Ivory Coast",
        "country": "Ivory Coast",
        "latency_ms_from_casablanca": 42.0,
        "compliance_frameworks": ["CI-DPA", "ISO 27001"],
    },
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=RegionListResponse)
async def list_regions(
    available_only: bool = Query(False, description="Show only available regions"),
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """List all available deployment regions with metadata.

    Region data is sourced from the hubs database — GPU counts, renewable
    percentages, and carbon intensities come from real hub records.
    Compliance frameworks and latency are regional metadata.

    Morocco is currently the only active region with 5 GPU hubs and
    1,798 GPUs. Other African regions are planned for future expansion.
    """
    # Query hubs from DB, grouped by country
    hubs_result = await db.execute(select(Hub))
    hubs = hubs_result.scalars().all()

    # Group hubs by country code
    hub_data_by_country: dict[str, dict] = {}
    # Proper ISO 3166-1 alpha-2 country code mapping
    COUNTRY_CODE_MAP = {
        "morocco": "MA",
        "algeria": "DZ",
        "tunisia": "TN",
        "egypt": "EG",
        "nigeria": "NG",
        "kenya": "KE",
        "south africa": "ZA",
        "senegal": "SN",
        "ghana": "GH",
        "ethiopia": "ET",
        "ivory coast": "CI",
        "côte d'ivoire": "CI",
        "cote d'ivoire": "CI",
    }
    for hub in hubs:
        # Derive country code from hub country — proper ISO mapping
        country = hub.country or "Morocco"
        code = COUNTRY_CODE_MAP.get(country.lower().strip(), country[:2].upper())
        if code not in hub_data_by_country:
            hub_data_by_country[code] = {
                "hub_count": 0,
                "total_gpus": 0,
                "available": False,
                "avg_renewable": [],
                "avg_carbon": [],
            }
        hub_data_by_country[code]["hub_count"] += 1
        hub_data_by_country[code]["total_gpus"] += hub.total_gpus or 0
        hub_data_by_country[code]["available"] = True
        hub_data_by_country[code]["avg_renewable"].append(hub.renewable_percentage or 0)
        hub_data_by_country[code]["avg_carbon"].append(hub.grid_carbon_intensity or 0)

    # Build regions list — always include all regions from metadata
    regions: list[RegionInfo] = []
    for code, meta in REGION_METADATA.items():
        hub_data = hub_data_by_country.get(code, {})
        is_available = hub_data.get("available", False)
        hub_count = hub_data.get("hub_count", 0)
        total_gpus = hub_data.get("total_gpus", 0)

        avg_renewable = 0.0
        if hub_data.get("avg_renewable"):
            avg_renewable = sum(hub_data["avg_renewable"]) / len(hub_data["avg_renewable"])

        avg_carbon = 0.0
        if hub_data.get("avg_carbon"):
            avg_carbon = sum(hub_data["avg_carbon"]) / len(hub_data["avg_carbon"])

        regions.append(RegionInfo(
            name=meta["name"],
            code=code,
            country=meta["country"],
            available=is_available,
            hub_count=hub_count,
            total_gpus=total_gpus,
            avg_renewable_percentage=round(avg_renewable, 2),
            avg_carbon_intensity=round(avg_carbon, 2),
            latency_ms_from_casablanca=meta["latency_ms_from_casablanca"],
            compliance_frameworks=meta["compliance_frameworks"],
        ))

    if available_only:
        regions = [r for r in regions if r.available]

    return RegionListResponse(
        regions=regions,
        total=len(regions),
    )
