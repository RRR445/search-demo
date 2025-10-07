"""FastAPI application exposing search and analytics endpoints backed by Elasticsearch."""

from __future__ import annotations

import contextlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, TransportError
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --- env ---
load_dotenv()
ES_URL = os.getenv("ELASTIC_URL")
ES_KEY = os.getenv("ELASTIC_API_KEY")
INDEX = os.getenv("ELASTIC_INDEX", "products")

SETTINGS_INDEX = "search_settings"
SETTINGS_ID = "default"


def _init_es_client() -> Elasticsearch | None:
    if not ES_URL or not ES_KEY:
        return None
    try:
        return Elasticsearch(ES_URL, api_key=ES_KEY)
    except ValueError:
        return None


es = _init_es_client()

# --- app ---
app = FastAPI(title="Search Demo API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: rajaa tuotantodomainiin
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)

# --- static UI ---
BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui.index")


@app.get("/index.html", response_class=HTMLResponse)
def serve_index() -> str:
    for candidate in (UI_DIR / "index.html", BASE_DIR / "index.html"):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return "<h1>index.html not found</h1>"


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


# --- indices (best-effort) ---
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
    with contextlib.suppress(Exception):
        es.indices.create(
            index=SETTINGS_INDEX,
            ignore=400,
            mappings={
                "properties": {
                    "weights": {"type": "object", "enabled": True},
                    "brand_boosts": {"type": "object", "enabled": True},
                    "updated_at": {"type": "date"},
                }
            },
        )


def _get_es_client() -> Elasticsearch:
    if es is None:
        raise HTTPException(status_code=503, detail="Search backend unavailable")
    return es


def _raise_service_unavailable(error: Exception) -> None:
    raise HTTPException(status_code=503, detail="Search backend unavailable") from error


# --- settings model & defaults ---
class Weights(BaseModel):
    rating: float = Field(0.4, ge=0, description="field_value_factor for rating (0..5)")
    reviews: float = Field(0.2, ge=0, description="weight for reviews (log1p)")
    sales_30d: float = Field(0.8, ge=0, description="weight for recent sales (log1p)")
    in_stock_bonus: float = Field(1.3, ge=0, description="extra weight when in_stock=True")
    newness_decay_days: int = Field(45, ge=1, description="gauss decay window in days")


class Settings(BaseModel):
    weights: Weights = Weights()
    brand_boosts: Dict[str, float] = Field(
        default_factory=lambda: {"purelux": 1.2, "autodude": 1.1}
    )


DEFAULT_SETTINGS = Settings()


def load_settings() -> Settings:
    try:
        doc = _get_es_client().get(index=SETTINGS_INDEX, id=SETTINGS_ID, ignore=[404])
        if doc and doc.get("found"):
            src = doc["_source"]
            return Settings(**src)
    except TransportError:
        pass
    return DEFAULT_SETTINGS


def save_settings(s: Settings) -> None:
    _get_es_client().index(
        index=SETTINGS_INDEX,
        id=SETTINGS_ID,
        document={**s.model_dump(), "updated_at": datetime.utcnow()},
        refresh="wait_for",
    )


# --- API models ---
class SearchResponse(BaseModel):
    total: int
    hits: list[dict[str, Any]]
    aggs: dict[str, Any] | None = None
    queryId: str | None = None


# --- routes ---
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


# Settings endpoints
@app.get("/settings", response_model=Settings)
def get_settings() -> Settings:
    return load_settings()


@app.post("/settings", response_model=Settings)
def update_settings(payload: Settings) -> Settings:
    # Tallennus ES:ään
    try:
        save_settings(payload)
        return payload
    except TransportError as error:
        _raise_service_unavailable(error)


# Suggest
@app.get("/suggest")
def suggest(q: str, size: int = 8) -> list[dict[str, Any]]:
    base_body = {
        "size": size,
        "_source": ["id", "sku", "name", "brand", "price", "in_stock", "image", "image_url"],
    }
    try:
        resp = _get_es_client().search(
            index=INDEX,
            body={**base_body, "query": {"match": {"name.ac": {"query": q, "operator": "and"}}}},
        )
    except TransportError as error:
        if getattr(error, "status_code", None) == 400:
            try:
                resp = _get_es_client().search(
                    index=INDEX,
                    body={**base_body, "query": {"match_phrase_prefix": {"name": {"query": q}}}},
                )
            except TransportError as fallback_error:
                _raise_service_unavailable(fallback_error)
        else:
            _raise_service_unavailable(error)

    return [hit["_source"] | {"_id": hit["_id"]} for hit in resp["hits"]["hits"]]


# Search that uses settings-based scoring
@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query("", description="Hakusana"),
    page: int = Query(1, ge=1),
    size: int = Query(24, ge=1, le=100),
    in_stock: Optional[bool] = Query(None),
    brand: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    sort: Optional[str] = Query(None, description="relevance | price_asc | price_desc | newest"),
) -> SearchResponse:
    s = load_settings()  # haetaan aina (voi myöhemmin cachettaa)

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

    # Build scoring functions from settings
    functions: list[dict[str, Any]] = [
        {"weight": s.weights.in_stock_bonus, "filter": {"term": {"in_stock": True}}},
        {
            "gauss": {
                "added_at": {
                    "origin": "now",
                    "scale": f"{s.weights.newness_decay_days}d",
                    "decay": 0.5,
                }
            }
        },
        # Rating 0..5 (no modifier)
        {
            "field_value_factor": {
                "field": "rating",
                "factor": s.weights.rating,
                "missing": 0.0,
            }
        },
        # Reviews -> log1p to normalize big counts
        {
            "field_value_factor": {
                "field": "reviews",
                "factor": s.weights.reviews,
                "modifier": "log1p",
                "missing": 0.0,
            }
        },
        # Recent sales (e.g. last 30d)
        {
            "field_value_factor": {
                "field": "sales_30d",
                "factor": s.weights.sales_30d,
                "modifier": "log1p",
                "missing": 0.0,
            }
        },
    ]

    # Brand boosts from settings (keyword field recommended)
    if s.brand_boosts:
        for bname, w in s.brand_boosts.items():
            try:
                w_float = float(w)
            except Exception:
                continue
            functions.append({"filter": {"term": {"brand": bname.lower()}}, "weight": w_float})

    body: dict[str, Any] = {
        "from": (page - 1) * size,
        "size": size,
        "query": {
            "function_score": {
                "query": {"bool": {"must": must or {"match_all": {}}, "filter": filters}},
                "score_mode": "sum",
                "boost_mode": "multiply",
                "functions": functions,
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
    aggs = {
        key: (value.get("buckets", value))
        for key, value in (resp.get("aggregations") or {}).items()
    }

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
