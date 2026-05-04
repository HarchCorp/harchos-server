"""Seed script – creates default user, API key, and 5 Moroccan hubs."""

import asyncio
import hashlib

from sqlalchemy import select

from app.database import async_session_factory, init_db
from app.models.user import User
from app.models.api_key import ApiKey
from app.models.hub import Hub
from app.config import settings


# Moroccan hub data
MOROCCAN_HUBS = [
    {
        "name": "Harch Alpha",
        "region": "Dakhla-Oued Ed-Dahab",
        "status": "ready",
        "tier": "enterprise",
        "total_gpus": 256,
        "available_gpus": 198,
        "total_cpu_cores": 2048,
        "available_cpu_cores": 1536,
        "total_memory_gb": 8192.0,
        "available_memory_gb": 6144.0,
        "total_storage_gb": 50000.0,
        "available_storage_gb": 35000.0,
        "latitude": 23.6848,
        "longitude": -15.9570,
        "city": "Dakhla",
        "country": "Morocco",
        "renewable_percentage": 92.5,
        "grid_carbon_intensity": 45.0,
        "pue": 1.08,
        "sovereignty_level": "strict",
        "data_residency_policy": "local_only",
    },
    {
        "name": "Harch Beta",
        "region": "Tanger-Tetouan-Al Hoceima",
        "status": "ready",
        "tier": "performance",
        "total_gpus": 128,
        "available_gpus": 96,
        "total_cpu_cores": 1024,
        "available_cpu_cores": 768,
        "total_memory_gb": 4096.0,
        "available_memory_gb": 3072.0,
        "total_storage_gb": 25000.0,
        "available_storage_gb": 18000.0,
        "latitude": 35.7595,
        "longitude": -5.8340,
        "city": "Tanger",
        "country": "Morocco",
        "renewable_percentage": 78.3,
        "grid_carbon_intensity": 120.0,
        "pue": 1.15,
        "sovereignty_level": "standard",
        "data_residency_policy": "local_only",
    },
    {
        "name": "Harch Gamma",
        "region": "Draa-Tafilalet",
        "status": "ready",
        "tier": "enterprise",
        "total_gpus": 512,
        "available_gpus": 384,
        "total_cpu_cores": 4096,
        "available_cpu_cores": 3072,
        "total_memory_gb": 16384.0,
        "available_memory_gb": 12288.0,
        "total_storage_gb": 100000.0,
        "available_storage_gb": 70000.0,
        "latitude": 30.9189,
        "longitude": -6.9000,
        "city": "Ouarzazate",
        "country": "Morocco",
        "renewable_percentage": 96.8,
        "grid_carbon_intensity": 25.0,
        "pue": 1.05,
        "sovereignty_level": "strict",
        "data_residency_policy": "local_only",
    },
    {
        "name": "Harch Delta",
        "region": "Casablanca-Settat",
        "status": "ready",
        "tier": "standard",
        "total_gpus": 64,
        "available_gpus": 32,
        "total_cpu_cores": 512,
        "available_cpu_cores": 256,
        "total_memory_gb": 2048.0,
        "available_memory_gb": 1024.0,
        "total_storage_gb": 10000.0,
        "available_storage_gb": 6000.0,
        "latitude": 33.5731,
        "longitude": -7.5898,
        "city": "Casablanca",
        "country": "Morocco",
        "renewable_percentage": 55.0,
        "grid_carbon_intensity": 200.0,
        "pue": 1.25,
        "sovereignty_level": "standard",
        "data_residency_policy": "regional",
    },
    {
        "name": "Harch Epsilon",
        "region": "Marrakech-Safi",
        "status": "ready",
        "tier": "performance",
        "total_gpus": 192,
        "available_gpus": 144,
        "total_cpu_cores": 1536,
        "available_cpu_cores": 1152,
        "total_memory_gb": 6144.0,
        "available_memory_gb": 4608.0,
        "total_storage_gb": 30000.0,
        "available_storage_gb": 20000.0,
        "latitude": 31.6295,
        "longitude": -8.0160,
        "city": "Benguerir",
        "country": "Morocco",
        "renewable_percentage": 85.0,
        "grid_carbon_intensity": 65.0,
        "pue": 1.10,
        "sovereignty_level": "standard",
        "data_residency_policy": "local_only",
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

        # Create Moroccan hubs
        for hub_data in MOROCCAN_HUBS:
            hub = Hub(**hub_data)
            session.add(hub)

        await session.commit()
        print(f"Seeded {len(MOROCCAN_HUBS)} Moroccan hubs:")
        for hub_data in MOROCCAN_HUBS:
            print(f"  - {hub_data['name']} ({hub_data['city']})")

        print("\nSeed complete!")


if __name__ == "__main__":
    asyncio.run(seed())
