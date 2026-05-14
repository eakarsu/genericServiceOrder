"""Per-sector embedding store + semantic menu search API.

Uses a lightweight TF-IDF-like keyword-vector fallback so the route works
without a vector DB. Wire to ChromaDB / pgvector when available.
"""
from __future__ import annotations

import math
import os
import re
import threading
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter(prefix="/api/semantic-menu", tags=["semantic-menu"])


class MenuItem(BaseModel):
    id: str
    sector: str
    name: str
    description: str = ""
    price: Optional[float] = None
    tags: Optional[List[str]] = None


class IngestReq(BaseModel):
    items: List[MenuItem]


class SearchReq(BaseModel):
    sector: str
    query: str
    top_k: int = 5


# In-memory store, keyed by sector.
_store: Dict[str, List[Dict]] = defaultdict(list)
_idf: Dict[str, Dict[str, float]] = defaultdict(dict)
_lock = threading.Lock()


def _tokens(s: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 1]


def _recompute_idf(sector: str) -> None:
    items = _store[sector]
    if not items:
        _idf[sector] = {}
        return
    df = Counter()
    for it in items:
        for t in set(it["_tokens"]):
            df[t] += 1
    N = len(items)
    _idf[sector] = {t: math.log((1 + N) / (1 + c)) + 1 for t, c in df.items()}


@router.post("/ingest")
async def ingest(req: IngestReq):
    with _lock:
        per_sector = defaultdict(list)
        for it in req.items:
            tokens = _tokens(f"{it.name} {it.description} {' '.join(it.tags or [])}")
            per_sector[it.sector].append({"item": it.dict(), "_tokens": tokens})
        for sector, rows in per_sector.items():
            _store[sector].extend(rows)
            _recompute_idf(sector)
    return {"ok": True, "ingested": len(req.items), "sectors": list({i.sector for i in req.items})}


@router.post("/search")
async def search(req: SearchReq):
    items = _store.get(req.sector)
    if not items:
        return {"hits": [], "note": f"no items for sector={req.sector}"}
    q_tokens = _tokens(req.query)
    q_vec = Counter(q_tokens)
    idf = _idf.get(req.sector, {})

    def score(item_tokens: List[str]) -> float:
        v = Counter(item_tokens)
        s = 0.0
        for t, qc in q_vec.items():
            if t in v:
                s += qc * v[t] * idf.get(t, 1.0)
        denom = math.sqrt(sum(c * c for c in v.values())) * math.sqrt(sum(c * c for c in q_vec.values()))
        return s / (denom or 1)

    scored = sorted(items, key=lambda x: score(x["_tokens"]), reverse=True)[: req.top_k]
    return {"hits": [{"score": round(score(s["_tokens"]), 4), "item": s["item"]} for s in scored]}


@router.get("/{sector}")
async def list_sector(sector: str):
    return {"count": len(_store.get(sector, [])), "items": [s["item"] for s in _store.get(sector, [])[:50]]}
