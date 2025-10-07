"""FastAPI application exposing search and analytics endpoints backed by Elasticsearch."""

from __future__ import annotations

import contextlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, TransportError
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


load_dotenv()


ES_URL = os.getenv("ELASTIC_URL")
ES_KEY = os.getenv("ELASTIC_API_KEY")
INDEX = os.getenv("ELASTIC_INDEX", "products")


def _init_es_client() -> Elasticsearch | None:
    """Create the Elasticsearch client when configuration is present."""
    if not ES_URL or not ES_KEY:
        return None
    try:
        return Elasticsearch(ES_URL, api_key=ES_KEY)
    except ValueError:
        return None


es = _init_es_client()


app = FastAPI(title="Search Demo API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: rajaa tuotantoon omaan domainiin
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui.index")


@app.get("/index.html", response_class=HTMLResponse)
def serve_index() -> str:
    """Serve index.html for UI."""
    for candidate in (UI_DIR / "index.html", BASE_DIR / "index.html"):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return "<h1>index.html not found</h1>"


@app.get("/favicon.ico")
def favicon() -> Response:
    """Silence favicon requests."""
    return Response(status_code=204)


if es is not None:
    with contextlib.suppress(Exception):
        es.indices.create(
            index="search_events",
            ignore=400,
            mappings={
                "properties": {
                    "ts": {"type": "date"},
                    "type": {"type": "keyword"},
                    "query": {"type": "keyword"},
                    "queryId": {"type": "keyword"},
                    "product_id": {"type": "keyword"},
                    "meta": {"type": "object", "enabled": False},
                }
            },
        )


class SearchResponse(BaseModel):
    total: int
    hits: list[dict[str, Any]]
    aggs: dict[str, Any] | None = None
    queryId: str | None = None


def _raise_service_unavailable(error: Exception) -> None:
    raise HTTPException(status_code=503, detail="Search backend unavailable") from error


def _get_es_client() -> Elasticsearch:
    if es is None:
        raise HTTPException(status_code=503, detail="Search backend unavailable")
    return es


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "search-demo", "version": app.version or "unknown"}


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        info = _get_es_client().info()
    except TransportError as error:
        _raise_service_unavailable(error)
    return {"ok": True, "es_version": info["version"]["number"]}


@app.get("/suggest")
def suggest(q: str, size: int = 8) -> list[dict[str, Any]]:
    base_body = {
        "size": size,
        "_source": ["id", "sku", "name", "brand", "price", "in_stock", "image", "image_url"],
    }
    try:
        resp = _get_es_client().search(
            index=INDEX,
            body={
                **base_body,
                "query": {"match": {"name.ac": {"query": q, "operator": "and"}}},
            },
        )
    except TransportError as error:
        if getattr(error, "status_code", None) == 400:
            try:
                resp = _get_es_client().search(
                    index=INDEX,
                    body={
                        **base_body,
                        "query": {"match_phrase_prefix": {"name": {"query": q}}},
                    },
                )
            except TransportError as fallback_error:
                _raise_service_unavailable(fallback_error)
        else:
            _raise_service_unavailable(error)

    return [hit["_source"] | {"_id": hit["_id"]} for hit in resp["hits"]["hits"]]


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query("", description="Hakusana"),
    page: int = Query(1, ge=1),
    size: int = Query(24, ge=1, le=100),
    in_stock: bool | None = Query(None),
    brand: str | None = Query(None),
    category: str | None = Query(None),
    min_price: float | None = Query(None),
    max_price: float | None = Query(None),
    sort: str | None = Query(None, description="relevance | price_asc | price_desc | newest"),
) -> SearchResponse:
    must: list[dict[str, Any]] = []
    if q:
        must.append(
            {
                "multi_match": {
                    "query": q,
                    "type": "best_fields",
                    "fields": ["name^10", "brand^8", "category_path^3", "description"],
                    "fuzziness": "AUTO",
                    "operator": "and",
                }
            }
        )

    filters: list[dict[str, Any]] = []
    if in_stock is not None:
        filters.append({"term": {"in_stock": in_stock}})
    if brand:
        filters.append({"term": {"brand": brand.lower()}})
    if category:
        filters.append({"term": {"category_path": category}})
    if min_price is not None or max_price is not None:
        price_range: dict[str, Any] = {}
        if min_price is not None:
            price_range["gte"] = min_price
        if max_price is not None:
            price_range["lte"] = max_price
        filters.append({"range": {"price": price_range}})

    sort_clause: list[dict[str, Any]] | None = None
    if sort == "price_asc":
        sort_clause = [{"price": "asc"}]
    elif sort == "price_desc":
        sort_clause = [{"price": "desc"}]
    elif sort == "newest":
        sort_clause = [{"added_at": "desc"}]

    body: dict[str, Any] = {
        "from": (page - 1) * size,
        "size": size,
        "query": {
            "function_score": {
                "query": {"bool": {"must": must or {"match_all": {}}, "filter": filters}},
                "score_mode": "sum",
                "boost_mode": "multiply",
                "functions": [
                    {"weight": 1.3, "filter": {"term": {"in_stock": True}}},
                    {"gauss": {"added_at": {"origin": "now", "scale": "45d", "decay": 0.5}}},
                ],
            }
        },
        "aggs": {
            "brand": {"terms": {"field": "brand", "size": 25}},
            "category": {"terms": {"field": "category_path", "size": 25}},
            "price_stats": {"stats": {"field": "price"}},
        },
        "_source": {"excludes": ["description"]},
    }
    if sort_clause:
        body["sort"] = sort_clause

    try:
        resp = _get_es_client().search(index=INDEX, body=body)
    except TransportError as error:
        _raise_service_unavailable(error)

    hits = [
        hit["_source"] | {"_id": hit["_id"], "_score": hit.get("_score")}
        for hit in resp["hits"]["hits"]
    ]
    total = resp["hits"]["total"]["value"]
    aggs = {key: (value.get("buckets", value)) for key, value in (resp.get("aggregations") or {}).items()}

    query_id = str(uuid4())
    with contextlib.suppress(TransportError):
        _get_es_client().index(
            index="search_events",
            document={"ts": datetime.utcnow(), "type": "search", "query": q, "queryId": query_id},
        )

    return SearchResponse(total=total, hits=hits, aggs=aggs, queryId=query_id)


@app.post("/click")
def click(product_id: str, queryId: str) -> dict[str, bool]:
    try:
        _get_es_client().index(
            index="search_events",
            document={"ts": datetime.utcnow(), "type": "click", "product_id": product_id, "queryId": queryId},
        )
    except TransportError as error:
        _raise_service_unavailable(error)
    return {"ok": True}
