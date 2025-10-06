import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from elasticsearch import Elasticsearch

# Env-muuttujat Renderin Environmentista
ES_URL  = os.getenv("ELASTIC_URL")
ES_KEY  = os.getenv("ELASTIC_API_KEY")
INDEX   = os.getenv("ELASTIC_INDEX", "products")

es = Elasticsearch(ES_URL, api_key=ES_KEY)
app = FastAPI(title="Search API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # rajaa omaan domainiin tuotannossa
    allow_credentials=False,
    allow_methods=["GET","POST"],
    allow_headers=["*"],
)

class SearchResponse(BaseModel):
    total: int
    hits: list
    aggs: dict | None = None

@app.get("/health")
def health():
    info = es.info()
    return {"ok": True, "es_version": info["version"]["number"]}

@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query("", description="Hakusana"),
    page: int = Query(1, ge=1),
    size: int = Query(24, ge=1, le=100),
    in_stock: bool | None = Query(None),
    brand: str | None = Query(None),
):
    must = []
    if q:
        must.append({
            "multi_match": {
                "query": q,
                "type": "best_fields",
                "fields": ["name^8","brand^5","category_path^3","description"]
            }
        })

    filters = []
    if in_stock is not None:
        filters.append({"term": {"in_stock": in_stock}})
    if brand:
        filters.append({"term": {"brand": brand.lower()}})

    body = {
      "from": (page-1)*size, "size": size,
      "query": {
        "function_score": {
          "query": {"bool": {"must": must or {"match_all": {}}, "filter": filters}},
          "score_mode": "sum", "boost_mode": "multiply",
          "functions": [
            {"weight": 1.3, "filter": {"term": {"in_stock": True}}},
            {"gauss": {"added_at": {"origin": "now", "scale": "45d", "decay": 0.5}}}
          ]
        }
      },
      "aggs": {
        "brand":    {"terms": {"field": "brand", "size": 25}},
        "category": {"terms": {"field": "category_path", "size": 25}}
      },
      "_source": {"excludes": ["description"]}
    }
    resp = es.search(index=INDEX, body=body)
    hits = [h["_source"] | {"_id": h["_id"], "_score": h["_score"]} for h in resp["hits"]["hits"]]
    total = resp["hits"]["total"]["value"]
    aggs  = {k: v["buckets"] for k, v in (resp.get("aggregations") or {}).items()}
    return {"total": total, "hits": hits, "aggs": aggs}
