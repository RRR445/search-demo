import os
from uuid import uuid4
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from elasticsearch import Elasticsearch, TransportError

# ---- Ympäristömuuttujat (asetat Renderissa) ----
ES_URL  = os.getenv("ELASTIC_URL")
ES_KEY  = os.getenv("ELASTIC_API_KEY")
INDEX   = os.getenv("ELASTIC_INDEX", "products")

# ---- Elasticsearch client ----
es = Elasticsearch(ES_URL, api_key=ES_KEY)

# ---- FastAPI app + CORS (rajaa tuotannossa omaan domainiin) ----
app = FastAPI(title="Search Demo API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # prod: ["https://www.oma-domain.fi"]
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---- Luo event-indeksi (CTR-lokit) jos puuttuu ----
try:
    es.indices.create(
        index="search_events",
        ignore=400,  # 400 = index_exists, ei kaadu
        mappings={
            "properties": {
                "ts":         {"type": "date"},
                "type":       {"type": "keyword"},     # "search" | "click"
                "query":      {"type": "keyword"},
                "queryId":    {"type": "keyword"},
                "product_id": {"type": "keyword"},
                "meta":       {"type": "object", "enabled": False},
            }
        },
    )
except Exception:
    pass

# ---- Malli vastaukselle ----
class SearchResponse(BaseModel):
    total: int
    hits: list
    aggs: Optional[dict] = None
    queryId: Optional[str] = None

# ---- Healthcheck ----
@app.get("/health")
def health():
    info = es.info()
    return {"ok": True, "es_version": info["version"]["number"]}

# ---- Autocomplete / suggest ----
@app.get("/suggest")
def suggest(q: str, size: int = 8):
    """
    Palauttaa nopeita ehdotuksia. Yrittää käyttää name.ac -kenttää (edge-ngram),
    ja jos sitä ei ole, fallbackina match_phrase_prefix name-kenttään.
    """
    try:
        body = {
            "size": size,
            "_source": ["id", "sku", "name", "brand", "price", "in_stock"],
            "query": {"match": {"name.ac": {"query": q, "operator": "and"}}},
        }
        resp = es.search(index=INDEX, body=body)
    except Exception:
        body = {
            "size": size,
            "_source": ["id", "sku", "name", "brand", "price", "in_stock"],
            "query": {"match_phrase_prefix": {"name": {"query": q}}},
        }
        resp = es.search(index=INDEX, body=body)

    return [h["_source"] | {"_id": h["_id"]} for h in resp["hits"]["hits"]]

# ---- Varsinainen haku facetien, hintafiltterin ja sorttauksen kanssa ----
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
    sort: Optional[str] = Query(None),  # "relevance" | "price_asc" | "price_desc" | "newest"
):
    must = []
    if q:
        must.append({
            "multi_match": {
                "query": q,
                "type": "best_fields",
                "fields": ["name^8", "brand^5", "category_path^3", "description"],
                "fuzziness": "AUTO"
            }
        })

    filters = []
    if in_stock is not None:
        filters.append({"term": {"in_stock": in_stock}})
    if brand:
        filters.append({"term": {"brand": brand.lower()}})
    if category:
        filters.append({"term": {"category_path": category}})
    if min_price is not None or max_price is not None:
        rng = {}
        if min_price is not None:
            rng["gte"] = min_price
        if max_price is not None:
            rng["lte"] = max_price
        filters.append({"range": {"price": rng}})

    sort_clause = None
    if sort == "price_asc":
        sort_clause = [{"price": "asc"}]
    elif sort == "price_desc":
        sort_clause = [{"price": "desc"}]
    elif sort == "newest":
        sort_clause = [{"added_at": "desc"}]
    # "relevance" oletus: ei sorttia -> järjestyy scorella

    body = {
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

    resp = es.search(index=INDEX, body=body)
    hits = [h["_source"] | {"_id": h["_id"], "_score": h["_score"]} for h in resp["hits"]["hits"]]
    total = resp["hits"]["total"]["value"]
    aggs = {k: (v.get("buckets", v)) for k, v in (resp.get("aggregations") or {}).items()}

    # Kirjaa hakutapahtuma CTR:ää varten
    query_id = str(uuid4())
    try:
        es.index(index="search_events", document={
            "ts": datetime.utcnow(),
            "type": "search",
            "query": q,
            "queryId": query_id
        })
    except TransportError:
        pass

    return {"total": total, "hits": hits, "aggs": aggs, "queryId": query_id}

# ---- Klikkiloki CTR-laskentaa varten ----
@app.post("/click")
def click(product_id: str, queryId: str):
    try:
        es.index(index="search_events", document={
            "ts": datetime.utcnow(),
            "type": "click",
            "product_id": product_id,
            "queryId": queryId
        })
        return {"ok": True}
    except TransportError as e:
        return {"ok": False, "error": str(e)}
