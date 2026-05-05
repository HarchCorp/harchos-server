"""Regions API endpoint.

Returns a catalog of available deployment regions across Africa with
metadata including hub counts, GPU capacity, renewable energy percentages,
carbon intensity, latency from Morocco, and compliance frameworks.
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

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
# Static region data
# ---------------------------------------------------------------------------

REGIONS: list[RegionInfo] = [
    RegionInfo(
        name="Morocco",
        code="MA",
        country="Morocco",
        available=True,
        hub_count=5,
        total_gpus=1798,
        avg_renewable_percentage=81.5,
        avg_carbon_intensity=47.0,
        latency_ms_from_casablanca=0.0,
        compliance_frameworks=[
            "ISO 27001",
            "GDPR",
            "CNDP (Moroccan DPA)",
            "Sovereign Cloud",
            "PCI DSS",
        ],
    ),
    RegionInfo(
        name="Algeria",
        code="DZ",
        country="Algeria",
        available=False,
        hub_count=0,
        total_gpus=0,
        avg_renewable_percentage=1.5,
        avg_carbon_intensity=420.0,
        latency_ms_from_casablanca=22.0,
        compliance_frameworks=[
            "ANDI",
            "GDPR (partial)",
        ],
    ),
    RegionInfo(
        name="Tunisia",
        code="TN",
        country="Tunisia",
        available=False,
        hub_count=0,
        total_gpus=0,
        avg_renewable_percentage=6.0,
        avg_carbon_intensity=380.0,
        latency_ms_from_casablanca=28.0,
        compliance_frameworks=[
            "INPDP (Tunisian DPA)",
            "GDPR (partial)",
        ],
    ),
    RegionInfo(
        name="Egypt",
        code="EG",
        country="Egypt",
        available=False,
        hub_count=0,
        total_gpus=0,
        avg_renewable_percentage=12.0,
        avg_carbon_intensity=350.0,
        latency_ms_from_casablanca=55.0,
        compliance_frameworks=[
            "EITDA",
            "PCI DSS",
            "ISO 27001",
        ],
    ),
    RegionInfo(
        name="Nigeria",
        code="NG",
        country="Nigeria",
        available=False,
        hub_count=0,
        total_gpus=0,
        avg_renewable_percentage=18.0,
        avg_carbon_intensity=310.0,
        latency_ms_from_casablanca=85.0,
        compliance_frameworks=[
            "NDPR",
            "NITDA",
            "ISO 27001",
        ],
    ),
    RegionInfo(
        name="Kenya",
        code="KE",
        country="Kenya",
        available=False,
        hub_count=0,
        total_gpus=0,
        avg_renewable_percentage=75.0,
        avg_carbon_intensity=120.0,
        latency_ms_from_casablanca=110.0,
        compliance_frameworks=[
            "DPA 2019",
            "ISO 27001",
            "PCI DSS",
        ],
    ),
    RegionInfo(
        name="South Africa",
        code="ZA",
        country="South Africa",
        available=False,
        hub_count=0,
        total_gpus=0,
        avg_renewable_percentage=8.0,
        avg_carbon_intensity=450.0,
        latency_ms_from_casablanca=145.0,
        compliance_frameworks=[
            "POPIA",
            "PCI DSS",
            "ISO 27001",
            "SOC 2",
        ],
    ),
    RegionInfo(
        name="Senegal",
        code="SN",
        country="Senegal",
        available=False,
        hub_count=0,
        total_gpus=0,
        avg_renewable_percentage=22.0,
        avg_carbon_intensity=280.0,
        latency_ms_from_casablanca=35.0,
        compliance_frameworks=[
            "ADP (Senegalese DPA)",
            "ISO 27001",
        ],
    ),
    RegionInfo(
        name="Ghana",
        code="GH",
        country="Ghana",
        available=False,
        hub_count=0,
        total_gpus=0,
        avg_renewable_percentage=30.0,
        avg_carbon_intensity=240.0,
        latency_ms_from_casablanca=50.0,
        compliance_frameworks=[
            "DPA 2012",
            "ISO 27001",
        ],
    ),
    RegionInfo(
        name="Ethiopia",
        code="ET",
        country="Ethiopia",
        available=False,
        hub_count=0,
        total_gpus=0,
        avg_renewable_percentage=90.0,
        avg_carbon_intensity=25.0,
        latency_ms_from_casablanca=95.0,
        compliance_frameworks=[
            "FDRE Proclamation",
        ],
    ),
    RegionInfo(
        name="Ivory Coast",
        code="CI",
        country="Ivory Coast",
        available=False,
        hub_count=0,
        total_gpus=0,
        avg_renewable_percentage=35.0,
        avg_carbon_intensity=260.0,
        latency_ms_from_casablanca=42.0,
        compliance_frameworks=[
            "CI-DPA",
            "ISO 27001",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=RegionListResponse)
async def list_regions(
    available_only: bool = Query(False, description="Show only available regions"),
):
    """List all available deployment regions with metadata.

    Returns region information including hub counts, total GPU capacity,
    renewable energy percentages, carbon intensity, latency from Casablanca
    (Morocco), and supported compliance frameworks.

    Morocco is currently the only active region with 5 GPU hubs and
    1,798 GPUs. Other African regions are planned for future expansion.
    """
    regions = REGIONS
    if available_only:
        regions = [r for r in regions if r.available]

    return RegionListResponse(
        regions=regions,
        total=len(regions),
    )
