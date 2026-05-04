"""Model service – CRUD operations for ML models."""

import json

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.model import Model
from app.schemas.model import (
    ModelCreate,
    ModelUpdate,
    ModelResponse,
    ModelMetadata,
    ModelSpecSchema,
    ModelCapabilities,
)
from app.schemas.common import PaginatedResponse

class ModelService:
    """Service for model CRUD operations."""

    @staticmethod
    def _to_response(m: Model) -> ModelResponse:
        """Convert an ORM Model to a ModelResponse schema with nested metadata/spec."""

        # Parse stored JSON fields
        capabilities = None
        if m.capabilities_json:
            try:
                capabilities = ModelCapabilities(**json.loads(m.capabilities_json))
            except (json.JSONDecodeError, Exception):
                capabilities = None

        # --- metadata ---
        metadata = ModelMetadata(
            id=m.id,
            name=m.name,
            created_at=m.created_at,
            updated_at=m.updated_at,
            labels={},
            annotations={},
        )

        # --- spec ---
        data_residency = None
        spec = ModelSpecSchema(
            name=m.name,
            framework=m.framework,
            task=m.task,
            version="1.0.0",
            description=None,
            hub_id=m.hub_id,
            sovereignty_level="strict",
            data_residency=data_residency,
            labels={},
            tags=[],
        )

        # --- top-level status ---
        # Map internal statuses to SDK ModelStatus values
        status_map = {
            "draft": "available",
            "training": "available",
            "ready": "available",
            "deployed": "deployed",
            "archived": "unavailable",
        }
        mapped_status = status_map.get(m.status, "available")

        return ModelResponse(
            metadata=metadata,
            spec=spec,
            status=mapped_status,
            framework=m.framework,
            task=m.task,
            size=None,
            capabilities=capabilities,
            deployed_at=None,
            inference_endpoint=None,
        )

    @staticmethod
    async def list_models(
        db: AsyncSession,
        page: int = 1,
        per_page: int = 20,
        framework: str | None = None,
        status: str | None = None,
        hub_id: str | None = None,
    ) -> PaginatedResponse[ModelResponse]:
        """List models with pagination and optional filters."""
        query = select(Model)
        count_query = select(func.count()).select_from(Model)

        if framework:
            query = query.where(Model.framework == framework)
            count_query = count_query.where(Model.framework == framework)
        if status:
            query = query.where(Model.status == status)
            count_query = count_query.where(Model.status == status)
        if hub_id:
            query = query.where(Model.hub_id == hub_id)
            count_query = count_query.where(Model.hub_id == hub_id)

        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        offset = (page - 1) * per_page
        query = query.offset(offset).limit(per_page).order_by(Model.created_at.desc())

        result = await db.execute(query)
        models = result.scalars().all()

        items = [ModelService._to_response(m) for m in models]
        return PaginatedResponse.create(items=items, total=total, page=page, per_page=per_page)

    @staticmethod
    async def get_model(db: AsyncSession, model_id: str) -> ModelResponse | None:
        """Get a single model by ID."""
        result = await db.execute(select(Model).where(Model.id == model_id))
        m = result.scalar_one_or_none()
        if not m:
            return None
        return ModelService._to_response(m)

    @staticmethod
    async def create_model(db: AsyncSession, data: ModelCreate) -> ModelResponse:
        """Create a new model."""
        # Serialize capabilities if provided
        capabilities_json = None
        if data.capabilities:
            capabilities_json = json.dumps(data.capabilities.model_dump() if hasattr(data.capabilities, 'model_dump') else data.capabilities)

        metrics_json = None
        if data.metrics:
            metrics_json = json.dumps(data.metrics)

        m = Model(
            name=data.name,
            framework=data.framework,
            task=data.task,
            status=data.status or "draft",
            capabilities_json=capabilities_json,
            metrics_json=metrics_json,
            hub_id=data.hub_id,
        )
        db.add(m)
        await db.flush()
        return ModelService._to_response(m)

    @staticmethod
    async def update_model(
        db: AsyncSession, model_id: str, data: ModelUpdate
    ) -> ModelResponse | None:
        """Update a model."""
        result = await db.execute(select(Model).where(Model.id == model_id))
        m = result.scalar_one_or_none()
        if not m:
            return None

        update_data = data.model_dump(exclude_unset=True)

        if "capabilities" in update_data:
            cap = update_data.pop("capabilities")
            m.capabilities_json = json.dumps(cap) if cap is not None else None

        if "metrics" in update_data:
            met = update_data.pop("metrics")
            m.metrics_json = json.dumps(met) if met is not None else None

        for field in ("name", "framework", "task", "status", "hub_id"):
            if field in update_data and update_data[field] is not None:
                setattr(m, field, update_data[field])

        await db.flush()
        return ModelService._to_response(m)

    @staticmethod
    async def delete_model(db: AsyncSession, model_id: str) -> bool:
        """Delete a model. Returns True if deleted, False if not found."""
        result = await db.execute(select(Model).where(Model.id == model_id))
        m = result.scalar_one_or_none()
        if not m:
            return False
        await db.delete(m)
        await db.flush()
        return True
