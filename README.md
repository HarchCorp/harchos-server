# HarchOS Server

The FastAPI + PostgreSQL backend for the HarchOS ecosystem.

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
curl -H "X-API-Key: hsk_test_development_key_12345" http://localhost:8000/v1/workloads
```

## Docker Compose (PostgreSQL)

```bash
docker compose up -d
```

## API Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health` | GET | Health check |
| `/v1/hubs` | GET | List hubs (public) |
| `/v1/hubs` | POST | Create hub (auth) |
| `/v1/hubs/{id}` | GET/ PATCH/ DELETE | Hub CRUD |
| `/v1/hubs/{id}/capacity` | GET | Hub capacity |
| `/v1/workloads` | GET/ POST | List/ Create workloads (auth) |
| `/v1/workloads/{id}` | GET/ PATCH/ DELETE | Workload CRUD |
| `/v1/models` | GET/ POST | List/ Create models (auth) |
| `/v1/models/{id}` | GET/ PATCH/ DELETE | Model CRUD |
| `/v1/energy/reports/{id}` | GET | Energy report |
| `/v1/energy/summary` | GET | Energy summary |
| `/v1/energy/green-windows` | GET | Green windows |
| `/v1/energy/consumption/{id}` | GET | Energy consumption |
| `/v1/auth/api-keys` | POST | Create API key |
| `/v1/auth/token` | POST | Exchange key for JWT |
| `/v1/auth/me` | GET | Current user info |

## Authentication

- **API Key**: Send via `X-API-Key` header or `Authorization: Bearer hsk_...`
- **JWT Token**: Exchange API key for token via `POST /v1/auth/token`, then use `Authorization: Bearer hst_...`
- **Default test key**: `hsk_test_development_key_12345`

## Seeded Data

5 Moroccan hubs:
- Harch Alpha (Dakhla) – Enterprise tier, 96.8% renewable
- Harch Beta (Tanger) – Performance tier, 78.3% renewable
- Harch Gamma (Ouarzazate) – Enterprise tier, 96.8% renewable
- Harch Delta (Casablanca) – Standard tier, 55% renewable
- Harch Epsilon (Benguerir) – Performance tier, 85% renewable
