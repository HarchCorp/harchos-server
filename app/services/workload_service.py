"""Workload service – CRUD operations for workloads."""

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.workload import Workload
from app.schemas.workload import (
    WorkloadCreate,
    WorkloadUpdate,
    WorkloadResponse,
    WorkloadMetadata,
    WorkloadSpecSchema,
    WorkloadCompute,
    DataResidencySpec,
)
from app.schemas.common import PaginatedResponse

class WorkloadService:
    """Service for workload CRUD operations."""

    @staticmethod
    def _to_response(wl: Workload) -> WorkloadResponse:
        """Convert an ORM Workload to a WorkloadResponse schema with nested metadata/spec."""

        # --- metadata ---
        metadata = WorkloadMetadata(
            id=wl.id,
            name=wl.name,
            created_at=wl.created_at,
            updated_at=wl.updated_at,
            resource_type="workload",
            labels={},
            annotations={},
        )

        # --- spec ---
        # Build DataResidency from the stored policy string
        if wl.data_residency_policy == "local_only":
            data_residency = DataResidencySpec(allowed_regions=["morocco"])
        elif wl.data_residency_policy:
            data_residency = DataResidencySpec(allowed_regions=["morocco"])
        else:
            data_residency = None

        compute = WorkloadCompute(
            gpu_count=wl.gpu_count,
            gpu_type=wl.gpu_type or None,
            cpu_cores=max(wl.cpu_cores, 1),
            memory_gb=max(wl.memory_gb, 1.0),
            storage_gb=max(wl.storage_gb, 10.0),
        )

        # Map internal sovereignty levels to SDK enum values (strict/moderate/minimal)
        sov_level_map = {"strict": "strict", "standard": "moderate", "minimal": "minimal"}
        mapped_sov_level = sov_level_map.get(wl.sovereignty_level, "moderate")

        spec = WorkloadSpecSchema(
            name=wl.name,
            type=wl.type,
            hub_id=wl.hub_id,
            compute=compute,
            priority=wl.priority,
            sovereignty_level=mapped_sov_level,
            data_residency=data_residency,
            labels={},
        )

        # --- carbon_metrics (optional) ---
        carbon_metrics = None

        return WorkloadResponse(
            metadata=metadata,
            spec=spec,
            status=wl.status,
            hub_id=wl.hub_id,
            carbon_metrics=carbon_metrics,
            started_at=wl.created_at if wl.status == "running" else None,
            completed_at=wl.updated_at if wl.status in ("completed", "failed", "cancelled") else None,
            error_message=None,
            retry_count=0,
        )

    @staticmethod
    async def list_workloads(
        db: AsyncSession,
        page: int = 1,
        per_page: int = 20,
        status: str | None = None,
        workload_type: str | None = None,
        hub_id: str | None = None,
    ) -> PaginatedResponse[WorkloadResponse]:
        """List workloads with pagination and optional filters."""
        query = select(Workload)
        count_query = select(func.count()).select_from(Workload)

        if status:
            query = query.where(Workload.status == status)
            count_query = count_query.where(Workload.status == status)
        if workload_type:
            query = query.where(Workload.type == workload_type)
            count_query = count_query.where(Workload.type == workload_type)
        if hub_id:
            query = query.where(Workload.hub_id == hub_id)
            count_query = count_query.where(Workload.hub_id == hub_id)

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination
        offset = (page - 1) * per_page
        query = query.offset(offset).limit(per_page).order_by(Workload.created_at.desc())

        result = await db.execute(query)
        workloads = result.scalars().all()

        items = [WorkloadService._to_response(wl) for wl in workloads]
        return PaginatedResponse.create(items=items, total=total, page=page, per_page=per_page)

    @staticmethod
    async def get_workload(db: AsyncSession, workload_id: str) -> WorkloadResponse | None:
        """Get a single workload by ID."""
        result = await db.execute(select(Workload).where(Workload.id == workload_id))
        wl = result.scalar_one_or_none()
        if not wl:
            return None
        return WorkloadService._to_response(wl)

    @staticmethod
    async def create_workload(db: AsyncSession, data: WorkloadCreate) -> WorkloadResponse:
        """Create a new workload."""
        # Support both new SDK format and legacy sovereignty format
        sovereignty_level = data.sovereignty_level
        data_residency_policy = ""
        carbon_aware = False
        carbon_intensity_threshold = None

        if data.sovereignty:
            sovereignty_level = data.sovereignty.level
            data_residency_policy = data.sovereignty.data_residency_policy
            carbon_aware = data.sovereignty.carbon_metrics.carbon_aware
            carbon_intensity_threshold = data.sovereignty.carbon_metrics.carbon_intensity_threshold

        if data.data_residency and data.data_residency.allowed_regions:
            data_residency_policy = "local_only"

        wl = Workload(
            name=data.name,
            type=data.type,
            status="pending",
            gpu_count=data.compute.gpu_count,
            gpu_type=data.compute.gpu_type or "",
            cpu_cores=data.compute.cpu_cores,
            memory_gb=data.compute.memory_gb,
            storage_gb=data.compute.storage_gb,
            hub_id=data.hub_id,
            priority=data.priority,
            sovereignty_level=sovereignty_level,
            data_residency_policy=data_residency_policy,
            carbon_aware=carbon_aware,
            carbon_intensity_threshold=carbon_intensity_threshold,
        )
        db.add(wl)
        await db.flush()
        return WorkloadService._to_response(wl)

    @staticmethod
    async def update_workload(
        db: AsyncSession, workload_id: str, data: WorkloadUpdate
    ) -> WorkloadResponse | None:
        """Update a workload."""
        result = await db.execute(select(Workload).where(Workload.id == workload_id))
        wl = result.scalar_one_or_none()
        if not wl:
            return None

        update_data = data.model_dump(exclude_unset=True)

        # Handle nested compute object
        if "compute" in update_data and update_data["compute"] is not None:
            compute = update_data.pop("compute")
            wl.gpu_count = compute.get("gpu_count", wl.gpu_count)
            wl.gpu_type = compute.get("gpu_type", wl.gpu_type)
            wl.cpu_cores = compute.get("cpu_cores", wl.cpu_cores)
            wl.memory_gb = compute.get("memory_gb", wl.memory_gb)
            wl.storage_gb = compute.get("storage_gb", wl.storage_gb)

        # Handle nested sovereignty object (legacy)
        if "sovereignty" in update_data and update_data["sovereignty"] is not None:
            sov = update_data.pop("sovereignty")
            wl.sovereignty_level = sov.get("level", wl.sovereignty_level)
            wl.data_residency_policy = sov.get("data_residency_policy", wl.data_residency_policy)
            if "carbon_metrics" in sov and sov["carbon_metrics"] is not None:
                cm = sov["carbon_metrics"]
                wl.carbon_aware = cm.get("carbon_aware", wl.carbon_aware)
                wl.carbon_intensity_threshold = cm.get("carbon_intensity_threshold", wl.carbon_intensity_threshold)

        # Handle simple fields
        for field in ("name", "type", "status", "hub_id", "priority", "sovereignty_level"):
            if field in update_data and update_data[field] is not None:
                setattr(wl, field, update_data[field])

        await db.flush()
        return WorkloadService._to_response(wl)

    @staticmethod
    async def delete_workload(db: AsyncSession, workload_id: str) -> bool:
        """Delete a workload. Returns True if deleted, False if not found."""
        result = await db.execute(select(Workload).where(Workload.id == workload_id))
        wl = result.scalar_one_or_none()
        if not wl:
            return False
        await db.delete(wl)
        await db.flush()
        return True
