# AI-Powered Real-Time Financial Crime and Money Muling Detection Engine

Educational decision-support prototype. It uses synthetic/demo data only; a detection is a risk signal for human review, not proof of criminal activity.

## Phase 1 status

This repository started as an empty Git repository with only this README. Phase 1 establishes a small Node.js, Express, and TypeScript backend with:

- `GET /health` for process health
- `GET /health/db` for an accurate PostgreSQL connectivity check
- request IDs, request logging, centralized errors, not-found handling, and graceful shutdown
- environment validation without hard-coded credentials
- automated health and error tests

Transaction persistence, migrations, graph analysis, AI services, realtime events, cases, and frontend features are intentionally deferred to later phases.

## Requirements

- Node.js 20 or newer
- npm 10 or newer
- PostgreSQL is optional for starting the backend, but required for a healthy `/health/db` response

## Run the backend

```powershell
cd backend
npm install
Copy-Item .env.example .env
npm run dev
```

The server listens on `http://localhost:3000` by default.

Check process health:

```powershell
Invoke-RestMethod http://localhost:3000/health
```

Check database health:

```powershell
Invoke-RestMethod http://localhost:3000/health/db
```

Without a configured or running PostgreSQL database, `/health/db` returns HTTP 503 and explains that the database is unavailable. The application does not pretend that this check passed.

## Test and build

```powershell
cd backend
npm test
npm run typecheck
npm run build
```

## Environment variables

Copy `backend/.env.example` to `backend/.env`. `.env` is ignored by Git.

| Variable | Default | Purpose |
| --- | --- | --- |
| `NODE_ENV` | `development` | Runtime environment |
| `PORT` | `3000` | HTTP port |
| `DATABASE_URL` | unset | PostgreSQL connection string |
| `DB_CONNECT_TIMEOUT_MS` | `3000` | Database health-check timeout |
| `LOG_LEVEL` | `info` | Request logging level |

## Next phases

After Phase 1 approval, Phase 2 will add PostgreSQL migrations, synthetic accounts, and validated/idempotent transaction ingestion. Later phases will add graph rules, explainable risk assessments, human-review cases, Python/NetworkX integration, Socket.IO events, frontend integration, and measured evaluation.
