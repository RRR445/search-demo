# search-demo

A minimal FastAPI + Elasticsearch example that powers autocomplete, product search, and click tracking endpoints. This template keeps the production deployment on [Render](https://render.com) simple while providing local tooling and CI so teams can iterate safely.

## API Overview

The application exposes four JSON endpoints backed by Elasticsearch and one informational root route:

| Method | Route      | Description |
| ------ | ---------- | ----------- |
| `GET`  | `/`        | Basic service metadata. |
| `GET`  | `/health`  | Confirms the API can reach Elasticsearch. |
| `GET`  | `/suggest` | Lightweight autocomplete query. |
| `GET`  | `/search`  | Faceted product search with filtering & scoring. |
| `POST` | `/click`   | Records click events tied to previous searches. |

All responses are JSON and require Elasticsearch credentials provided via environment variables.

## Local Development

```bash
python -m venv .venv && source .venv/bin/activate
make install
cp .env.example .env   # fill values
make dev               # http://127.0.0.1:8080/health
```

FastAPI's test client depends on `httpx`. Install it once (`pip install httpx`) before running `make test` locally.

The development server reloads on changes. For production-like behavior locally, use `make run` to bind on all interfaces without reload.

### Environment Variables

| Name | Description |
| ---- | ----------- |
| `ELASTIC_URL` | HTTPS endpoint for your Elasticsearch cluster (e.g. `https://example.es.amazonaws.com:443`). |
| `ELASTIC_API_KEY` | Elasticsearch API key with read/write access to the configured indices. |
| `ELASTIC_INDEX` | Product index used for search and suggest queries. Defaults to `products`. |

Create a `.env` file locally (based on `.env.example`) or configure these variables in your hosting provider.

## Running the API with Docker

```bash
docker build -t search-demo .
docker run --rm -p 8080:10000 \
  -e ELASTIC_URL=... -e ELASTIC_API_KEY=... -e ELASTIC_INDEX=products \
  search-demo
```

Then open `http://127.0.0.1:8080/health`.

## Render Deployment

1. Connect this repository to Render and select **New Web Service**.
2. Render reads the provided `render.yaml` and builds a Docker service automatically on every push to the default branch.
3. In the Render dashboard, add the environment variables `ELASTIC_URL`, `ELASTIC_API_KEY`, and (optionally) `ELASTIC_INDEX` via the **Environment** tab.
4. Deploy — health checks are available at `/health`.

## Branch Protection Guidance

Enable branch protection on `main` in GitHub:

1. Navigate to **Settings → Branches → Branch protection rules**.
2. Add a rule for `main`.
3. Require pull request reviews before merging (at least 1 approving review).
4. Require status checks to pass before merging and select the CI workflow from this repository.
5. Optional: enforce admins for stronger guarantees.

## API Examples

Set `BASE` to your deployment URL (e.g. `https://search-demo.onrender.com`).

```bash
curl "$BASE/health"

curl "$BASE/suggest?q=vah"

curl "$BASE/search?q=painepesuri&in_stock=true&min_price=50&max_price=400&sort=price_asc"

curl -X POST "$BASE/click?product_id=SKU123&queryId=UUID"
```

Happy searching!
