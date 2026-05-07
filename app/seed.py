"""Seed script – creates default user, API key, optimized Moroccan hubs, and pricing plans."""

import asyncio
import hashlib
import logging

from sqlalchemy import select, func

from app.database import async_session_factory, init_db
from app.models.user import User
from app.models.api_key import ApiKey
from app.models.hub import Hub
from app.models.pricing import Pricing, BillingRecord
from app.models.model import Model
from app.config import settings
from app.services.auth_service import AuthService

logger = logging.getLogger("harchos.seed")


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
        existing_user = result.scalar_one_or_none()
        if existing_user:
            # Fix: Ensure the admin user has role='admin' (may have been seeded
            # with role='user' in earlier versions that didn't pass role explicitly)
            if existing_user.email == "admin@harchos.ai" and existing_user.role != "admin":
                existing_user.role = "admin"
                logger.info("Updated admin user role from '%s' to 'admin'", existing_user.role)
                await session.flush()

            # Fix: Ensure the admin API key has tier='enterprise'
            key_result = await session.execute(
                select(ApiKey).where(ApiKey.user_id == existing_user.id, ApiKey.is_active.is_(True))
            )
            admin_key = key_result.scalar_one_or_none()
            if admin_key and admin_key.tier != "enterprise":
                admin_key.tier = "enterprise"
                logger.info("Updated admin API key tier from '%s' to 'enterprise'", admin_key.tier)
                await session.flush()

            logger.info("Database already seeded. Skipping.")
            return

        # Create default user (admin role for full platform access)
        user = User(
            email="admin@harchos.ai",
            name="HarchOS Admin",
            is_active=True,
            role="admin",
        )
        session.add(user)
        await session.flush()

        # Create default API key
        # Use a fixed key for predictable testing if no env override is set
        test_key = settings.default_api_key or "hsk_prod_e2e_test_2026_morocco"
        key_hash = hashlib.sha256(test_key.encode()).hexdigest()
        key_prefix = test_key[:8]

        api_key = ApiKey(
            user_id=user.id,
            name="Production Admin Key",
            tier="enterprise",
            key_hash=key_hash,
            key_prefix=key_prefix,
            is_active=True,
        )
        session.add(api_key)
        await session.flush()

        logger.info("Created default user: %s", user.email)
        logger.info("Created test API key: %s***", test_key[:8])

        # Create optimized Moroccan hubs
        for hub_data in MOROCCAN_HUBS:
            hub = Hub(**hub_data)
            session.add(hub)

        await session.flush()
        logger.info("Seeded %d Moroccan hubs:", len(MOROCCAN_HUBS))
        total_gpus = 0
        for hub_data in MOROCCAN_HUBS:
            total_gpus += hub_data["total_gpus"]
            logger.info("  - %s (%s): %d GPUs, %.1f%% renewable, %.0f gCO2/kWh", hub_data['name'], hub_data['city'], hub_data['total_gpus'], hub_data['renewable_percentage'], hub_data['grid_carbon_intensity'])
        logger.info("  Total: %d GPUs across %d hubs", total_gpus, len(MOROCCAN_HUBS))

        # Create pricing plans
        for plan_data in PRICING_PLANS:
            plan = Pricing(**plan_data)
            session.add(plan)

        # --- Seed AI Models ---
        model_count_result = await session.execute(select(func.count()).select_from(Model))
        model_count = model_count_result.scalar() or 0
        if model_count == 0:
            logger.info("Seeding AI models...")
            # Fetch hubs for model assignment
            hubs_result = await session.execute(select(Hub).limit(1))
            first_hub = hubs_result.scalar_one_or_none()
            ai_models = [
                {"name": "Llama 3.3 70B Instruct", "framework": "llama", "task": "text-generation", "status": "ready"},
                {"name": "Llama 3.3 8B Instruct", "framework": "llama", "task": "text-generation", "status": "ready"},
                {"name": "Llama 4 Maverick 17Bx128E", "framework": "llama", "task": "text-generation", "status": "ready"},
                {"name": "Mistral Large 2411", "framework": "mistral", "task": "text-generation", "status": "ready"},
                {"name": "Mistral Small 2501", "framework": "mistral", "task": "text-generation", "status": "ready"},
                {"name": "Qwen 2.5 72B Instruct", "framework": "qwen", "task": "text-generation", "status": "ready"},
                {"name": "Qwen 2.5 7B Instruct", "framework": "qwen", "task": "text-generation", "status": "ready"},
                {"name": "DeepSeek V3", "framework": "deepseek", "task": "text-generation", "status": "ready"},
                {"name": "DeepSeek R1 70B", "framework": "deepseek", "task": "reasoning", "status": "ready"},
                {"name": "Gemma 3 27B IT", "framework": "gemma", "task": "text-generation", "status": "ready"},
                {"name": "Gemma 3 4B IT", "framework": "gemma", "task": "text-generation", "status": "ready"},
                {"name": "Phi-4 14B", "framework": "phi", "task": "text-generation", "status": "ready"},
                {"name": "CodeGemma 7B", "framework": "gemma", "task": "code-generation", "status": "ready"},
                {"name": "StarCoder2 15B", "framework": "starcoder", "task": "code-generation", "status": "ready"},
                {"name": "Command R 35B", "framework": "cohere", "task": "text-generation", "status": "ready"},
                {"name": "Command R+ 104B", "framework": "cohere", "task": "text-generation", "status": "ready"},
                {"name": "Mixtral 8x22B Instruct", "framework": "mistral", "task": "text-generation", "status": "ready"},
                {"name": "Mixtral 8x7B Instruct", "framework": "mistral", "task": "text-generation", "status": "ready"},
                {"name": "Yi 1.5 34B Chat", "framework": "yi", "task": "text-generation", "status": "ready"},
                {"name": "SOLAR 10.7B Instruct", "framework": "solar", "task": "text-generation", "status": "ready"},
                {"name": "Llama Guard 4 12B", "framework": "llama", "task": "safety", "status": "ready"},
                {"name": "HarchOS Embedding 3 Large", "framework": "embedding", "task": "embeddings", "status": "ready"},
            ]
            for model_data in ai_models:
                model = Model(
                    name=model_data["name"],
                    framework=model_data["framework"],
                    task=model_data["task"],
                    status=model_data["status"],
                    hub_id=first_hub.id if first_hub else None,
                )
                session.add(model)
            await session.flush()
            logger.info("Seeded %d AI models", len(ai_models))

        await session.commit()
        logger.info("Seeded %d pricing plans:", len(PRICING_PLANS))
        for plan_data in PRICING_PLANS:
            logger.info("  - %s (%s): $%.2f/gpu-hr in %s", plan_data['name'], plan_data['currency'], plan_data['price_per_gpu_hour'], plan_data['region'])

        logger.info("Seed complete!")


if __name__ == "__main__":
    asyncio.run(seed())
