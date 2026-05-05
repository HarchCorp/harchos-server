# HarchOS Server

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-D22128?logo=apache&logoColor=white)](https://www.apache.org/licenses/LICENSE-2.0)
[![GitHub](https://img.shields.io/badge/GitHub-HarchCorp-181717?logo=github&logoColor=white)](https://github.com/HarchCorp)

The **carbon-aware, sovereignty-first** GPU orchestration platform by HarchCorp.
Built on Morocco's renewable energy advantage to deliver the greenest AI compute on the planet.

**30+ API endpoints** across 10 endpoint groups.

## ✨ Feature Highlights

| Feature | Description |
|---------|-------------|
| 🌍 **Carbon-Aware Scheduling** | Automatically routes workloads to the greenest GPU hub based on real-time carbon intensity data |
| 🔒 **Data Sovereignty** | Strict data residency controls with local-only storage policies and sovereign cloud compliance |
| 🌐 **Multi-Region** | 5 Moroccan GPU hubs optimized for carbon intensity, with Pan-African expansion planned |
| 💰 **Tiered Pricing** | Enterprise, Performance, and Standard tiers with transparent billing in USD, MAD, and EUR |
| ⚡ **1,798 GPUs** | Carbon-optimized distribution: Ouarzazate (800) → Dakhla (400) → Benguerir (350) → Tanger (200) → Casablanca (48) |
| 🌱 **~47 gCO2/kWh Avg** | Average grid carbon intensity across all hubs (vs 91 before optimization) |

## Quick Start (Local/SQLite)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# In another terminal, seed the database
python -m app.seed

# Test endpoints
curl http://localhost:8000/v1/health
curl http://localhost:8000/v1/hubs
curl http://localhost:8000/v1/regions
curl http://localhost:8000/v1/pricing/plans
curl http://localhost:8000/v1/monitoring/metrics
curl -H "X-API-Key: hsk_test_development_key_12345" http://localhost:8000/v1/workloads
```

## Docker Compose (PostgreSQL)

### Development

```bash
docker compose up -d
```

### Production

```bash
docker compose -f docker-compose.production.yml up -d
```

See [docker-compose.production.yml](docker-compose.production.yml) for PostgreSQL 16 + Redis 7 setup.

## API Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| **Health** | | |
| `/v1/health` | GET | Health check |
| `/v1/monitoring/metrics` | GET | Platform-wide metrics |
| `/v1/monitoring/health/detailed` | GET | Detailed health check |
| **Auth** | | |
| `/v1/auth/api-keys` | POST | Create API key |
| `/v1/auth/token` | POST | Exchange key for JWT |
| `/v1/auth/me` | GET | Current user info |
| **Hubs** | | |
| `/v1/hubs` | GET/ POST | List/ Create hubs |
| `/v1/hubs/{id}` | GET/ PATCH/ DELETE | Hub CRUD |
| `/v1/hubs/{id}/capacity` | GET | Hub capacity |
| **Workloads** | | |
| `/v1/workloads` | GET/ POST | List/ Create workloads (auth) |
| `/v1/workloads/{id}` | GET/ PATCH/ DELETE | Workload CRUD |
| **Models** | | |
| `/v1/models` | GET/ POST | List/ Create models (auth) |
| `/v1/models/{id}` | GET/ PATCH/ DELETE | Model CRUD |
| **Energy** | | |
| `/v1/energy/reports/{id}` | GET | Energy report |
| `/v1/energy/summary` | GET | Energy summary |
| `/v1/energy/green-windows` | GET | Green windows |
| `/v1/energy/consumption/{id}` | GET | Energy consumption |
| **Carbon** | | |
| `/v1/carbon/intensity/{zone}` | GET | Zone carbon intensity |
| `/v1/carbon/intensity` | GET | All zone intensities |
| `/v1/carbon/optimal-hub` | POST | Find optimal hub |
| `/v1/carbon/optimize` | POST | Carbon-aware optimization |
| `/v1/carbon/forecast/{zone}` | GET | Carbon forecast |
| `/v1/carbon/metrics` | GET | Carbon metrics |
| `/v1/carbon/dashboard` | GET | Carbon dashboard |
| **Pricing** | | |
| `/v1/pricing/plans` | GET | List pricing plans |
| `/v1/pricing/plans/{id}` | GET | Get pricing plan |
| `/v1/pricing/estimate` | GET | Cost estimate |
| `/v1/pricing/billing/records` | GET | Billing records (auth) |
| `/v1/pricing/billing/records/{id}` | GET | Billing record (auth) |
| **Regions** | | |
| `/v1/regions` | GET | List deployment regions |
| **Monitoring** | | |
| `/v1/monitoring/metrics` | GET | Platform metrics |
| `/v1/monitoring/health/detailed` | GET | Detailed health |

## Authentication

- **API Key**: Send via `X-API-Key` header or `Authorization: Bearer hsk_...`
- **JWT Token**: Exchange API key for token via `POST /v1/auth/token`, then use `Authorization: Bearer hst_...`
- **Default test key**: `hsk_test_development_key_12345`

## Seeded Data

### GPU Hubs (Carbon-Optimized Distribution)

| Hub | City | Tier | GPUs | Renewable | Carbon Intensity |
|-----|------|------|------|-----------|------------------|
| Harch Ouarzazate | Ouarzazate | Enterprise | 800 | 97.2% | 18 gCO2/kWh |
| Harch Dakhla | Dakhla | Enterprise | 400 | 94.8% | 32 gCO2/kWh |
| Harch Benguerir | Benguerir | Performance | 350 | 88.5% | 55 gCO2/kWh |
| Harch Tanger | Tanger | Performance | 200 | 82.1% | 95 gCO2/kWh |
| Harch Casablanca | Casablanca | Standard | 48 | 45.0% | 210 gCO2/kWh |
| **Total** | | | **1,798** | **81.5% avg** | **~47 avg** |

### Pricing Plans

| Plan | GPU Type | Region | Price (USD) | Price (MAD) |
|------|----------|--------|-------------|-------------|
| H100 Enterprise | H100 | Ouarzazate | $2.10/gpu-hr | 21.00 MAD/gpu-hr |
| H100 Performance | H100 | Benguerir | $2.35/gpu-hr | — |
| A100 Performance | A100 | Tanger | $1.80/gpu-hr | — |
| A100 Standard | A100 | Casablanca | $1.95/gpu-hr | — |
| L40S Enterprise | L40S | Dakhla | $1.40/gpu-hr | — |
| L40S Performance | L40S | Benguerir | $1.55/gpu-hr | — |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HARCHOS_APP_NAME` | HarchOS Server | Application name |
| `HARCHOS_APP_VERSION` | 0.2.0 | Application version |
| `HARCHOS_DEBUG` | true | Enable debug mode |
| `HARCHOS_ENVIRONMENT` | dev | Environment: dev/staging/production |
| `HARCHOS_LOG_LEVEL` | INFO | Logging level |
| `HARCHOS_DATABASE_URL` | sqlite+aiosqlite:///./harchos.db | Database connection URL |
| `HARCHOS_SECRET_KEY` | harchos-dev-... | Secret key for JWT signing |
| `HARCHOS_CORS_ORIGINS` | ["*"] | Allowed CORS origins |
| `HARCHOS_RATE_LIMIT_REQUESTS_PER_MINUTE` | 60 | Rate limit per minute |
| `HARCHOS_REDIS_URL` | (empty) | Redis URL for caching (optional) |
| `HARCHOS_CARBON_GREEN_THRESHOLD_GCO2_KWH` | 200.0 | Carbon intensity green threshold |
| `HARCHOS_CARBON_CACHE_TTL_MINUTES` | 30 | Carbon data cache TTL |
| `HARCHOS_ELECTRICITY_MAPS_API_KEY` | (empty) | Electricity Maps API key |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        HarchOS Server                          │
│                     FastAPI + SQLAlchemy                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Health   │  │   Auth   │  │  Hubs    │  │  Workloads    │  │
│  │ /health   │  │ /auth/*  │  │ /hubs/*  │  │ /workloads/*  │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Models  │  │  Energy  │  │  Carbon  │  │   Pricing     │  │
│  │ /models/* │  │/energy/* │  │/carbon/* │  │ /pricing/*    │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
│                                                                 │
│  ┌──────────┐  ┌──────────────┐                                 │
│  │ Regions  │  │  Monitoring  │                                 │
│  │/regions/*│  │/monitoring/* │                                 │
│  └──────────┘  └──────────────┘                                 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                      Services Layer                             │
│  AuthService · HubService · WorkloadService · CarbonService    │
│  ModelService · EnergyService                                   │
├─────────────────────────────────────────────────────────────────┤
│                      Data Layer                                 │
│  ┌─────────────┐  ┌─────────┐  ┌───────────┐                   │
│  │ PostgreSQL   │  │  Redis  │  │  SQLite   │                   │
│  │ (production) │  │ (cache) │  │   (dev)   │                   │
│  └─────────────┘  └─────────┘  └───────────┘                   │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Moroccan GPU Hubs                             │
│                                                                 │
│  🌞 Ouarzazate (800 GPUs)    🌊 Dakhla (400 GPUs)             │
│     97.2% renewable              94.8% renewable               │
│     18 gCO2/kWh                  32 gCO2/kWh                   │
│                                                                 │
│  🏔️ Benguerir (350 GPUs)     ⛵ Tanger (200 GPUs)             │
│     88.5% renewable              82.1% renewable               │
│     55 gCO2/kWh                  95 gCO2/kWh                   │
│                                                                 │
│  🏙️ Casablanca (48 GPUs)                                      │
│     45.0% renewable · 210 gCO2/kWh                             │
└─────────────────────────────────────────────────────────────────┘
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
