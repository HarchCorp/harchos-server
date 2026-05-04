"""Hub service – CRUD operations for hubs."""

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.hub import Hub
from app.schemas.hub import (
    HubCreate,
    HubUpdate,
    HubResponse,
    HubMetadata,
    HubSpec,
    HubCapacity,
    HubCarbonMetrics,
    DataResidencySpec,
)
from app.schemas.common import PaginatedResponse

class HubService:
    """Service for hub CRUD operations."""

    @staticmethod
    def _to_response(hub: Hub) -> HubResponse:
        """Convert an ORM Hub to a HubResponse schema with nested metadata/spec."""

        # --- metadata ---
        metadata = HubMetadata(
            id=hub.id,
            name=hub.name,
            created_at=hub.created_at,
            updated_at=hub.updated_at,
            labels={},
            annotations={},
        )

        # --- spec ---
        # Derive DataResidency from the stored policy string
        if hub.data_residency_policy == "local_only":
            data_residency = DataResidencySpec(
                allowed_regions=["morocco"],
            )
        elif hub.data_residency_policy == "regional":
            data_residency = DataResidencySpec(
                allowed_regions=["morocco", "north_africa"],
            )
        else:
            data_residency = None

        # Map internal sovereignty levels to SDK enum values (strict/moderate/minimal)
        sov_level_map = {"strict": "strict", "standard": "moderate", "minimal": "minimal"}
        mapped_sov_level = sov_level_map.get(hub.sovereignty_level, "moderate")

        spec = HubSpec(
            name=hub.name,
            region=hub.region,
            tier=hub.tier,
            sovereignty_level=mapped_sov_level,
            data_residency=data_residency,
            gpu_types=["h100", "a100"] if hub.tier in ("enterprise", "performance") else ["a100"],
            auto_scale=hub.tier != "starter",
            min_gpu_count=0,
            max_gpu_count=hub.total_gpus if hub.total_gpus > 0 else 8,
            carbon_aware_scheduling=hub.sovereignty_level == "strict",
            labels={},
        )

        # --- capacity ---
        capacity = HubCapacity(
            total_gpus=hub.total_gpus,
            available_gpus=hub.available_gpus,
            total_cpu_cores=hub.total_cpu_cores,
            available_cpu_cores=hub.available_cpu_cores,
            total_memory_gb=hub.total_memory_gb,
            available_memory_gb=hub.available_memory_gb,
            total_storage_gb=hub.total_storage_gb,
            available_storage_gb=hub.available_storage_gb,
        )

        # --- carbon_metrics ---
        # Compute reasonable carbon metrics from hub energy data
        energy_kwh = hub.total_gpus * 12.5  # rough estimate
        co2_grams = energy_kwh * (1 - hub.renewable_percentage / 100) * hub.grid_carbon_intensity
        carbon_metrics = HubCarbonMetrics(
            co2_grams=round(co2_grams, 2),
            energy_kwh=round(energy_kwh, 2),
            pue=hub.pue,
            region_grid_intensity=hub.grid_carbon_intensity,
            renewable_percentage=hub.renewable_percentage,
            measured_at=hub.updated_at or hub.created_at,
        )

        # --- endpoint ---
        region_slug = hub.region.lower().replace(" ", "-")
        endpoint = f"https://{region_slug}.harchos.io"

        return HubResponse(
            metadata=metadata,
            spec=spec,
            status=hub.status,
            capacity=capacity,
            carbon_metrics=carbon_metrics,
            endpoint=endpoint,
            active_workloads=0,
        )

    @staticmethod
    async def list_hubs(
        db: AsyncSession,
        page: int = 1,
        per_page: int = 20,
        status: str | None = None,
        tier: str | None = None,
        region: str | None = None,
    ) -> PaginatedResponse[HubResponse]:
        """List hubs with pagination and optional filters."""
        query = select(Hub)
        count_query = select(func.count()).select_from(Hub)

        if status:
            query = query.where(Hub.status == status)
            count_query = count_query.where(Hub.status == status)
        if tier:
            query = query.where(Hub.tier == tier)
            count_query = count_query.where(Hub.tier == tier)
        if region:
            query = query.where(Hub.region.ilike(f"%{region}%"))
            count_query = count_query.where(Hub.region.ilike(f"%{region}%"))

        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        offset = (page - 1) * per_page
        query = query.offset(offset).limit(per_page).order_by(Hub.created_at.desc())

        result = await db.execute(query)
        hubs = result.scalars().all()

        items = [HubService._to_response(hub) for hub in hubs]
        return PaginatedResponse.create(items=items, total=total, page=page, per_page=per_page)

    @staticmethod
    async def get_hub(db: AsyncSession, hub_id: str) -> HubResponse | None:
        """Get a single hub by ID."""
        result = await db.execute(select(Hub).where(Hub.id == hub_id))
        hub = result.scalar_one_or_none()
        if not hub:
            return None
        return HubService._to_response(hub)

    @staticmethod
    async def create_hub(db: AsyncSession, data: HubCreate) -> HubResponse:
        """Create a new hub."""
        # Determine capacity from either new-style or legacy fields
        if data.capacity:
            total_gpus = data.capacity.total_gpus
            available_gpus = data.capacity.available_gpus
            total_cpu_cores = data.capacity.total_cpu_cores
            available_cpu_cores = data.capacity.available_cpu_cores
            total_memory_gb = data.capacity.total_memory_gb
            available_memory_gb = data.capacity.available_memory_gb
            total_storage_gb = data.capacity.total_storage_gb
            available_storage_gb = data.capacity.available_storage_gb
        else:
            total_gpus = 0
            available_gpus = 0
            total_cpu_cores = 0
            available_cpu_cores = 0
            total_memory_gb = 0.0
            available_memory_gb = 0.0
            total_storage_gb = 50000.0
            available_storage_gb = 35000.0

        # Determine location
        latitude = data.location.latitude if data.location else 0.0
        longitude = data.location.longitude if data.location else 0.0
        city = data.location.city if data.location else ""
        country = data.location.country if data.location else ""

        # Determine energy
        renewable_percentage = data.energy.renewable_percentage if data.energy else 0.0
        grid_carbon_intensity = data.energy.grid_carbon_intensity if data.energy else 0.0
        pue = data.energy.pue if data.energy else 1.0

        # Determine sovereignty
        sovereignty_level = data.sovereignty_level
        data_residency_policy = data.data_residency_policy or "local_only"

        hub = Hub(
            name=data.name,
            region=data.region,
            status="creating",
            tier=data.tier,
            total_gpus=total_gpus,
            available_gpus=available_gpus,
            total_cpu_cores=total_cpu_cores,
            available_cpu_cores=available_cpu_cores,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            total_storage_gb=total_storage_gb,
            available_storage_gb=available_storage_gb,
            latitude=latitude,
            longitude=longitude,
            city=city,
            country=country,
            renewable_percentage=renewable_percentage,
            grid_carbon_intensity=grid_carbon_intensity,
            pue=pue,
            sovereignty_level=sovereignty_level,
            data_residency_policy=data_residency_policy,
        )
        db.add(hub)
        await db.flush()
        return HubService._to_response(hub)

    @staticmethod
    async def update_hub(
        db: AsyncSession, hub_id: str, data: HubUpdate
    ) -> HubResponse | None:
        """Update a hub."""
        result = await db.execute(select(Hub).where(Hub.id == hub_id))
        hub = result.scalar_one_or_none()
        if not hub:
            return None

        update_data = data.model_dump(exclude_unset=True)

        # Handle nested capacity
        if "capacity" in update_data and update_data["capacity"] is not None:
            cap = update_data.pop("capacity")
            hub.total_gpus = cap.get("total_gpus", hub.total_gpus)
            hub.available_gpus = cap.get("available_gpus", hub.available_gpus)
            hub.total_cpu_cores = cap.get("total_cpu_cores", hub.total_cpu_cores)
            hub.available_cpu_cores = cap.get("available_cpu_cores", hub.available_cpu_cores)
            hub.total_memory_gb = cap.get("total_memory_gb", hub.total_memory_gb)
            hub.available_memory_gb = cap.get("available_memory_gb", hub.available_memory_gb)
            hub.total_storage_gb = cap.get("total_storage_gb", hub.total_storage_gb)
            hub.available_storage_gb = cap.get("available_storage_gb", hub.available_storage_gb)

        # Handle nested location
        if "location" in update_data and update_data["location"] is not None:
            loc = update_data.pop("location")
            hub.latitude = loc.get("latitude", hub.latitude)
            hub.longitude = loc.get("longitude", hub.longitude)
            hub.city = loc.get("city", hub.city)
            hub.country = loc.get("country", hub.country)

        # Handle nested energy
        if "energy" in update_data and update_data["energy"] is not None:
            en = update_data.pop("energy")
            hub.renewable_percentage = en.get("renewable_percentage", hub.renewable_percentage)
            hub.grid_carbon_intensity = en.get("grid_carbon_intensity", hub.grid_carbon_intensity)
            hub.pue = en.get("pue", hub.pue)

        # Handle simple fields
        for field in ("name", "region", "status", "tier", "sovereignty_level", "data_residency_policy"):
            if field in update_data and update_data[field] is not None:
                setattr(hub, field, update_data[field])

        await db.flush()
        return HubService._to_response(hub)

    @staticmethod
    async def delete_hub(db: AsyncSession, hub_id: str) -> bool:
        """Delete a hub. Returns True if deleted, False if not found."""
        result = await db.execute(select(Hub).where(Hub.id == hub_id))
        hub = result.scalar_one_or_none()
        if not hub:
            return False
        await db.delete(hub)
        await db.flush()
        return True

    @staticmethod
    async def get_hub_capacity(db: AsyncSession, hub_id: str) -> HubCapacity | None:
        """Get capacity info for a specific hub."""
        result = await db.execute(select(Hub).where(Hub.id == hub_id))
        hub = result.scalar_one_or_none()
        if not hub:
            return None
        return HubCapacity(
            total_gpus=hub.total_gpus,
            available_gpus=hub.available_gpus,
            total_cpu_cores=hub.total_cpu_cores,
            available_cpu_cores=hub.available_cpu_cores,
            total_memory_gb=hub.total_memory_gb,
            available_memory_gb=hub.available_memory_gb,
            total_storage_gb=hub.total_storage_gb,
            available_storage_gb=hub.available_storage_gb,
        )
