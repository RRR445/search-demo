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
app = FastAPI(title="Search Demo API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: rajaa tuotantoon
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
                    "field_names": {"type": "object", "enabled": True},
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
    rating: float = Field(0.4, ge=0)
    reviews: float = Field(0.2, ge=0)
    sales_30d: float = Field(0.8, ge=0)
    in_stock_bonus: float = Field(1.3, ge=0)
    newness_decay_days: int = Field(45, ge=1)


class FieldNames(BaseModel):
    rating_field: str | None = "rating"
    reviews_field: str | None = "reviews"
    sales30_field: str | None = "sales_30d"
    added_at_field: str | None = "added_at"
    brand_field: str | None = "brand"
    category_field: str | None = "category_path"
    in_stock_field: str | None = "in_stock"
    price_field: str | None = "price"
    image_field: str | None = "image"
    image_url_field: str | None = "image_url"


class Settings(BaseModel):
    weights: Weights = Weights()
    brand_boosts: Dict[str, float] = Field(
        default_factory=lambda: {"purelux": 1.2, "autodude": 1.1}
    )
    field_names: FieldNames = FieldNames()


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
    try:
        save_settings(payload)
        return payload
    except TransportError as error:
        _raise_service_unavailable(error)


# Suggest
@app.get("/suggest")
def suggest(q: str, size: int = 8) -> list[dict[str, Any]]:
    s = load_settings()
    fn = s.field_names

    # Build _source fields based on configured names
    src_fields = {
        "id",
        "sku",
        fn.brand_field or "brand",
        fn.price_field or "price",
        fn.image_field or "image",
        fn.image_url_field or "image_url",
        "name",
        fn.in_stock_field or "in_stock",
    }

    base_body = {
        "size": size,
        "_source": [f for f in src_fields if f],
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

    # Attach _id; UI voi muodostaa kuvat kenttien mukaan
    return [hit["_source"] | {"_id": hit["_id"]} for hit in resp["hits"]["hits"]]


# Search that uses settings-based scoring and field names
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
    s = load_settings()
    fn = s.field_names

    # Query
    must: list[dict[str, Any]] = []
    if q:
        must.append(
            {
                "multi_match": {
                    "query": q,
                    "type": "best_fields",
                    "fields": ["name^10", (fn.brand_field or "brand") + "^8", (fn.category_field or "category_path") + "^3", "description"],
                    "fuzziness": "AUTO",
                    "operator": "and",
                }
            }
        )

    # Filters
    filters: list[dict[str, Any]] = []
    if in_stock is not None and fn.in_stock_field:
        filters.append({"term": {fn.in_stock_field: in_stock}})
    if brand and fn.brand_field:
        filters.append({"term": {fn.brand_field: brand.lower()}})
    if category and fn.category_field:
        filters.append({"term": {fn.category_field: category}})
    if (min_price is not None or max_price is not None) and fn.price_field:
        price_range: dict[str, Any] = {}
        if min_price is not None:
            price_range["gte"] = min_price
        if max_price is not None:
            price_range["lte"] = max_price
        filters.append({"range": {fn.price_field: price_range}})

    # Sort
    sort_clause: list[dict[str, Any]] | None = None
    if sort == "price_asc" and fn.price_field:
        sort_clause = [{fn.price_field: "asc"}]
    elif sort == "price_desc" and fn.price_field:
        sort_clause = [{fn.price_field: "desc"}]
    elif sort == "newest" and fn.added_at_field:
        sort_clause = [{fn.added_at_field: "desc"}]

    # Functions based on configured field names
    functions: list[dict[str, Any]] = []

    if fn.in_stock_field and s.weights.in_stock_bonus > 0:
        functions.append({"weight": s.weights.in_stock_bonus, "filter": {"term": {fn.in_stock_field: True}}})

    if fn.added_at_field:
        functions.append(
            {
                "gauss": {
                    fn.added_at_field: {
                        "origin": "now",
                        "scale": f"{s.weights.newness_decay_days}d",
                        "decay": 0.5,
                    }
                }
            }
        )

    if fn.rating_field:
        functions.append(
            {"field_value_factor": {"field": fn.rating_field, "factor": s.weights.rating, "missing": 0.0}}
        )
    if fn.reviews_field:
        functions.append(
            {
                "field_value_factor": {
                    "field": fn.reviews_field,
                    "factor": s.weights.reviews,
                    "modifier": "log1p",
                    "missing": 0.0,
                }
            }
        )
    if fn.sales30_field:
        functions.append(
            {
                "field_value_factor": {
                    "field": fn.sales30_field,
                    "factor": s.weights.sales_30d,
                    "modifier": "log1p",
                    "missing": 0.0,
                }
            }
        )

    # Brand boosts
    if s.brand_boosts and fn.brand_field:
        for bname, w in s.brand_boosts.items():
            try:
                w_float = float(w)
            except Exception:
                continue
            functions.append({"filter": {"term": {fn.brand_field: bname.lower()}}, "weight": w_float})

    # Aggregations using configured fields (fallback if missing)
    brand_field = fn.brand_field or "brand"
    category_field = fn.category_field or "category_path"

    body: dict[str, Any] = {
        "from": (page - 1) * size,
        "size": size,
        "query": {
            "function_score": {
                "query": {"bool": {"must": must or {"match_all": {}}, "filter": filters}},
                "score_mode": "sum",
                "boost_mode": "multiply",
                "functions": functions or [{"weight": 1.0}],
            }
        },
        "aggs": {
            "brand": {"terms": {"field": brand_field, "size": 25}},
            "category": {"terms": {"field": category_field, "size": 25}},
            "price_stats": {"stats": {"field": fn.price_field or "price"}},
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
