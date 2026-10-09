# Money-Mulling-Crime-Detection-engine

## Frontend

React, TypeScript, and Vite frontend for the financial-crime monitoring application.

### Run locally

```sh
npm install
npm run dev
```

Create a local `.env` based on `.env.example` and set `VITE_API_BASE_URL` to the backend origin. Do not add credentials or secrets to Vite environment variables; client-side values are public.

### Integration status

The responsive navigation and dashboard shell are in place. The dashboard intentionally displays unavailable states rather than synthetic or demo values. The other workspace routes identify that their data/actions are not connected.

`src/lib/apiClient.ts` provides the centralized request helper and reads the backend origin from `VITE_API_BASE_URL`. It accepts a path from a caller but deliberately defines no endpoints. Before wiring any view, obtain and document the approved endpoint paths, JSON shapes, status values, Socket.IO event names, graph detection evidence, and permitted case actions from the backend owners.

Shared frontend domain types are in `src/types/api.ts`. Align them with the confirmed contract before using them to decode API responses.

### Scripts

- `npm run dev` — start the Vite development server.
- `npm run build` — run the TypeScript project build and create production assets.
- `npm run lint` — run Oxlint.
- `npm run preview` — serve the production build locally.