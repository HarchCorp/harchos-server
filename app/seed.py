"""Seed script – creates default user, API key, optimized Moroccan hubs, and pricing plans."""

import asyncio
import hashlib

from sqlalchemy import select

from app.database import async_session_factory, init_db
from app.models.user import User
from app.models.api_key import ApiKey
from app.models.hub import Hub
from app.models.pricing import Pricing, BillingRecord
from app.config import settings


# ---------------------------------------------------------------------------
# Optimized Moroccan hub data (carbon-intensity-based distribution)
#
# Logic: Ouarzazate and Dakhla have the best solar/wind resources and lowest
# carbon intensity, so they get the most GPUs. Casablanca has the worst
# carbon intensity and receives minimal GPUs (latency-sensitive workloads only).
#
# Result: 1,798 total GPUs (vs 1,152 before) with average carbon intensity
# ~47 gCO2/kWh (vs ~91 before).
# ---------------------------------------------------------------------------
MOROCCAN_HUBS = [
    {
        "name": "Harch Ouarzazate",
        "region": "Draa-Tafilalet",
        "status": "ready",
        "tier": "enterprise",
        "total_gpus": 800,
        "available_gpus": 640,
        "total_cpu_cores": 6400,
        "available_cpu_cores": 5120,
        "total_memory_gb": 25600.0,
        "available_memory_gb": 20480.0,
        "total_storage_gb": 200000.0,
        "available_storage_gb": 150000.0,
        "latitude": 30.9189,
        "longitude": -6.9000,
        "city": "Ouarzazate",
        "country": "Morocco",
        "renewable_percentage": 97.2,
        "grid_carbon_intensity": 18.0,
        "pue": 1.04,
        "sovereignty_level": "strict",
        "data_residency_policy": "local_only",
    },
    {
        "name": "Harch Dakhla",
        "region": "Dakhla-Oued Ed-Dahab",
        "status": "ready",
        "tier": "enterprise",
        "total_gpus": 400,
        "available_gpus": 320,
        "total_cpu_cores": 3200,
        "available_cpu_cores": 2560,
        "total_memory_gb": 12800.0,
        "available_memory_gb": 10240.0,
        "total_storage_gb": 80000.0,
        "available_storage_gb": 60000.0,
        "latitude": 23.6848,
        "longitude": -15.9570,
        "city": "Dakhla",
        "country": "Morocco",
        "renewable_percentage": 94.8,
        "grid_carbon_intensity": 32.0,
        "pue": 1.06,
        "sovereignty_level": "strict",
        "data_residency_policy": "local_only",
    },
    {
        "name": "Harch Benguerir",
        "region": "Marrakech-Safi",
        "status": "ready",
        "tier": "performance",
        "total_gpus": 350,
        "available_gpus": 280,
        "total_cpu_cores": 2800,
        "available_cpu_cores": 2240,
        "total_memory_gb": 11200.0,
        "available_memory_gb": 8960.0,
        "total_storage_gb": 70000.0,
        "available_storage_gb": 50000.0,
        "latitude": 31.6295,
        "longitude": -8.0160,
        "city": "Benguerir",
        "country": "Morocco",
        "renewable_percentage": 88.5,
        "grid_carbon_intensity": 55.0,
        "pue": 1.09,
        "sovereignty_level": "standard",
        "data_residency_policy": "local_only",
    },
    {
        "name": "Harch Tanger",
        "region": "Tanger-Tetouan-Al Hoceima",
        "status": "ready",
        "tier": "performance",
        "total_gpus": 200,
        "available_gpus": 160,
        "total_cpu_cores": 1600,
        "available_cpu_cores": 1280,
        "total_memory_gb": 6400.0,
        "available_memory_gb": 5120.0,
        "total_storage_gb": 40000.0,
        "available_storage_gb": 28000.0,
        "latitude": 35.7595,
        "longitude": -5.8340,
        "city": "Tanger",
        "country": "Morocco",
        "renewable_percentage": 82.1,
        "grid_carbon_intensity": 95.0,
        "pue": 1.12,
        "sovereignty_level": "standard",
        "data_residency_policy": "local_only",
    },
    {
        "name": "Harch Casablanca",
        "region": "Casablanca-Settat",
        "status": "ready",
        "tier": "standard",
        "total_gpus": 48,
        "available_gpus": 32,
        "total_cpu_cores": 384,
        "available_cpu_cores": 256,
        "total_memory_gb": 1536.0,
        "available_memory_gb": 1024.0,
        "total_storage_gb": 15000.0,
        "available_storage_gb": 10000.0,
        "latitude": 33.5731,
        "longitude": -7.5898,
        "city": "Casablanca",
        "country": "Morocco",
        "renewable_percentage": 45.0,
        "grid_carbon_intensity": 210.0,
        "pue": 1.30,
        "sovereignty_level": "standard",
        "data_residency_policy": "regional",
    },
]


# ---------------------------------------------------------------------------
# Pricing plans (1 USD ≈ 10 MAD)
# ---------------------------------------------------------------------------
PRICING_PLANS = [
    {
        "name": "H100 Enterprise",
        "gpu_type": "H100",
        "price_per_gpu_hour": 2.10,
        "price_per_cpu_core_hour": 0.035,
        "price_per_gb_storage_month": 0.08,
        "price_per_gb_memory_hour": 0.006,
        "currency": "USD",
        "region": "Draa-Tafilalet",
        "tier": "enterprise",
        "is_default": True,
    },
    {
        "name": "H100 Enterprise (MAD)",
        "gpu_type": "H100",
        "price_per_gpu_hour": 21.00,
        "price_per_cpu_core_hour": 0.35,
        "price_per_gb_storage_month": 0.80,
        "price_per_gb_memory_hour": 0.06,
        "currency": "MAD",
        "region": "Draa-Tafilalet",
        "tier": "enterprise",
        "is_default": False,
    },
    {
        "name": "H100 Performance",
        "gpu_type": "H100",
        "price_per_gpu_hour": 2.35,
        "price_per_cpu_core_hour": 0.039,
        "price_per_gb_storage_month": 0.09,
        "price_per_gb_memory_hour": 0.007,
        "currency": "USD",
        "region": "Marrakech-Safi",
        "tier": "performance",
        "is_default": True,
    },
    {
        "name": "A100 Performance",
        "gpu_type": "A100",
        "price_per_gpu_hour": 1.80,
        "price_per_cpu_core_hour": 0.030,
        "price_per_gb_storage_month": 0.07,
        "price_per_gb_memory_hour": 0.005,
        "currency": "USD",
        "region": "Tanger-Tetouan-Al Hoceima",
        "tier": "performance",
        "is_default": True,
    },
    {
        "name": "A100 Standard",
        "gpu_type": "A100",
        "price_per_gpu_hour": 1.95,
        "price_per_cpu_core_hour": 0.032,
        "price_per_gb_storage_month": 0.075,
        "price_per_gb_memory_hour": 0.0055,
        "currency": "USD",
        "region": "Casablanca-Settat",
        "tier": "standard",
        "is_default": True,
    },
    {
        "name": "L40S Enterprise",
        "gpu_type": "L40S",
        "price_per_gpu_hour": 1.40,
        "price_per_cpu_core_hour": 0.023,
        "price_per_gb_storage_month": 0.06,
        "price_per_gb_memory_hour": 0.004,
        "currency": "USD",
        "region": "Dakhla-Oued Ed-Dahab",
        "tier": "enterprise",
        "is_default": True,
    },
    {
        "name": "L40S Performance",
        "gpu_type": "L40S",
        "price_per_gpu_hour": 1.55,
        "price_per_cpu_core_hour": 0.026,
        "price_per_gb_storage_month": 0.065,
        "price_per_gb_memory_hour": 0.0045,
        "currency": "USD",
        "region": "Marrakech-Safi",
        "tier": "performance",
        "is_default": True,
    },
]


async def seed():
    """Seed the database with initial data."""
    # Create tables
    await init_db()

    async with async_session_factory() as session:
        # Check if already seeded
        result = await session.execute(select(User).limit(1))
        if result.scalar_one_or_none():
            print("Database already seeded. Skipping.")
            return

        # Create default user
        user = User(
            email="admin@harchos.ai",
            name="HarchOS Admin",
            is_active=True,
        )
        session.add(user)
        await session.flush()

        # Create default API key
        test_key = settings.default_api_key
        key_hash = hashlib.sha256(test_key.encode()).hexdigest()
        key_prefix = test_key[:8]

        api_key = ApiKey(
            user_id=user.id,
            name="Development Test Key",
            key_hash=key_hash,
            key_prefix=key_prefix,
            is_active=True,
        )
        session.add(api_key)
        await session.flush()

        print(f"Created default user: {user.email}")
        print(f"Created test API key: {test_key}")

        # Create optimized Moroccan hubs
        for hub_data in MOROCCAN_HUBS:
            hub = Hub(**hub_data)
            session.add(hub)

        await session.flush()
        print(f"Seeded {len(MOROCCAN_HUBS)} Moroccan hubs:")
        total_gpus = 0
        for hub_data in MOROCCAN_HUBS:
            total_gpus += hub_data["total_gpus"]
            print(f"  - {hub_data['name']} ({hub_data['city']}): {hub_data['total_gpus']} GPUs, {hub_data['renewable_percentage']}% renewable, {hub_data['grid_carbon_intensity']} gCO2/kWh")
        print(f"  Total: {total_gpus} GPUs across {len(MOROCCAN_HUBS)} hubs")

        # Create pricing plans
        for plan_data in PRICING_PLANS:
            plan = Pricing(**plan_data)
            session.add(plan)

        await session.commit()
        print(f"\nSeeded {len(PRICING_PLANS)} pricing plans:")
        for plan_data in PRICING_PLANS:
            print(f"  - {plan_data['name']} ({plan_data['currency']}): ${plan_data['price_per_gpu_hour']}/gpu-hr in {plan_data['region']}")

        print("\nSeed complete!")


if __name__ == "__main__":
    asyncio.run(seed())
