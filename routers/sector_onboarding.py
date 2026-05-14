"""Self-serve onboarding for new verticals — drop a menu file, get indexed.

Accepts a multipart upload (CSV or JSON) and a sector slug. Files are indexed
into the in-memory semantic_menu store via the same shared module.
"""
from __future__ import annotations

import csv
import io
import json
import os
from typing import Any, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel


router = APIRouter(prefix="/api/sector-onboarding", tags=["sector-onboarding"])


@router.post("/upload")
async def upload(sector: str = Form(...), file: UploadFile = File(...)):
    if not sector or not file.filename:
        raise HTTPException(status_code=400, detail="sector and file required")
    data = await file.read()
    items: List[Dict[str, Any]] = []

    if file.filename.lower().endswith(".csv"):
        text = data.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(text))
        for i, row in enumerate(reader):
            items.append({
                "id": row.get("id") or f"{sector}-{i}",
                "sector": sector,
                "name": row.get("name", ""),
                "description": row.get("description", ""),
                "price": float(row["price"]) if row.get("price") else None,
                "tags": [t.strip() for t in (row.get("tags") or "").split(",") if t.strip()],
            })
    else:
        try:
            payload = json.loads(data.decode("utf-8", errors="ignore"))
            arr = payload if isinstance(payload, list) else payload.get("items", [])
            for i, row in enumerate(arr):
                items.append({
                    "id": row.get("id") or f"{sector}-{i}",
                    "sector": sector,
                    "name": row.get("name", ""),
                    "description": row.get("description", ""),
                    "price": row.get("price"),
                    "tags": row.get("tags") or [],
                })
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"json parse failed: {e}")

    # Push into the semantic store via direct import.
    try:
        from .semantic_menu import _store, _recompute_idf, _tokens, _lock  # type: ignore
        with _lock:
            for it in items:
                tokens = _tokens(f"{it['name']} {it['description']} {' '.join(it.get('tags') or [])}")
                _store[sector].append({"item": it, "_tokens": tokens})
            _recompute_idf(sector)
    except Exception as e:
        # Non-fatal — still report ingest count.
        return {"ok": False, "items": len(items), "warning": str(e)}

    return {"ok": True, "sector": sector, "ingested": len(items)}


@router.get("/sectors")
async def list_sectors():
    try:
        from .semantic_menu import _store  # type: ignore
        return {"sectors": [{"slug": k, "items": len(v)} for k, v in _store.items()]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
