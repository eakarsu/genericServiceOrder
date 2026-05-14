# Audit Note — GenericOrderingService

## Bucket
**HAS_CODE_NO_NODE_BACKEND (Python project)** — NO CODE CHANGES.

## Why no scaffold
The playbook scaffolds **Node/Express** backends. This project is a Python
application (LangChain + ChromaDB + LangGraph) and already has working LLM
integration. Adding a parallel Node backend would create two competing service
layers in the same repo.

## Initial state
- Source files (.js/.ts/.tsx/.jsx/.py): 91
- LLM-reference scan hits (legitimate, not detector false positives):
  - `updated_universal_service_bot.py` — uses `langchain_openai.ChatOpenAI`
    against an OpenAI-compatible endpoint
  - `test_updated_universal_service_bot.py`
- Layout:
  - Python entrypoints: `run.py`, `amazon_main.py`, `eleven_labs_main.py`,
    `admin_app.py`, `orderProcessor.py`, `menuIndexerIntentCalc.py`,
    `parseLLMIndexFiles.py`, `visualize_graph.py`,
    `updated_universal_service_bot.py`
  - `sectors/` — 10+ vertical subdomains (auto_repair, beauty_salon, healthcare,
    insurance, financial_services, food_delivery, etc.)
  - `prompts/`, `sector_prompts/`, `sector_rules/`
  - `frontend/` — React frontend (out of scope per playbook)
  - `Dockerfile`, `requirements.txt`, `README.md`
- Real-domain name: yes (Generic Ordering Service across verticals).

## Audit-report context
`/_AUDIT/reports/batch_10.md` §7 calls this SKELETON with "frontend exists; no
backend routes". That is **inaccurate** — there is a substantial Python backend
with LangChain-based AI; it just is not Node/Express. The report appears to
have only counted JS/TS routes.

## False-positive note (LLM scan)
The two .py hits are genuine LLM usages, **not** false positives. They use
`langchain_openai.ChatOpenAI` with a configurable base URL — i.e., effectively
the same OpenRouter-style integration the playbook prescribes, just via the
LangChain abstraction.

## Genuine gaps from the audit report
- No HTTP API surface that an external Node/JS client could hit (the Python
  bot is invoked via CLI / scripts).
- No Salesforce-style CRUD endpoints (orders, items, sectors, users).
- No persistence layer beyond ChromaDB vector store.

## Recommendation
- **Do not scaffold a competing Node backend.**
- If a Node API surface is required, place it under a clearly named subdir
  (e.g., `node-api/`) and bridge it to the existing Python services rather than
  re-implementing.
- Update the audit report to reclassify from SKELETON to "HAS_PYTHON_AI_BACKEND".

## Audit recommendations applied this batch

**None.** The Python LangChain/LangGraph + ChromaDB architecture is fully
out of scope for a Node-focused apply pass. The genuine gaps listed above
(no HTTP API surface, no CRUD endpoints, no persistence beyond Chroma) all
require either a fresh FastAPI scaffold or migrating the existing scripts
into a service layer — both substantive design decisions, not mechanical
work.

The sibling `GenericOrderingServiceEnhanced` project does have a partial
FastAPI backend (`backend/app/`); it would make more sense to graduate
this project's logic into that layout than to build a new one here.

## Backlog (deferred, prioritised)

1. **Decide whether to merge into `GenericOrderingServiceEnhanced`** —
   the enhanced sibling already has FastAPI + auth + users router scaffold.
   Consolidating avoids two parallel codebases. NEEDS-PRODUCT-DECISION.
2. **If kept standalone: scaffold a FastAPI `app.py`** — wrap
   `updated_universal_service_bot.py` and `orderProcessor.py` behind
   POST endpoints; ~50 LOC. PRE-REQ for everything below.
3. **CRUD endpoints** for `sectors/`, orders, menu items, users —
   depends on a Postgres/SQLite schema (no persistence today beyond
   the Chroma vector store).
4. **Standardise LLM provider** — `updated_universal_service_bot.py`
   uses `langchain_openai.ChatOpenAI` against a configurable base URL;
   align with sibling repos by switching to direct OpenRouter
   (matches `investment/`, `librelane/`, `makepdf/`, `pos/` patterns).
5. **Worker / queue layer** for long-running ingestion jobs — the
   `parseLLMIndexFiles.py` and `menuIndexerIntentCalc.py` scripts run
   synchronously today.

## Files touched
None.

## Apply pass 3 (frontend)

- **Status:** SKIPPED-NO-DOMAIN.
- Backend is Python LangChain/LangGraph + ChromaDB; only HTTP endpoints exposed are Twilio webhooks (`/voice`, `/sms`, `/token`, `/status` in `amazon_main.py` and `eleven_labs_main.py`). There is no general AI HTTP API surface to wire to.
- Frontend (`frontend/`) is a generic admin shell that calls `/api/orders`, `/api/users`, `/api/sectors` — none of which exist on the Python backend.
- Adding an AI page on the FE is gated on first scaffolding a FastAPI HTTP layer (backlog item #2 in this note). Out-of-scope for a mechanical FE pass.
- No files changed.
- Log: `/Users/erolakarsu/projects/_AUDIT/apply3_logs/ab3_61.md`.

## Apply pass 3 (Group B — FastAPI bootstrap)

This pass directly resolves the Apply-pass-3 (frontend) blocker above and
backlog item #2 ("scaffold a FastAPI `app.py`").

### What was scaffolded
- **`app.py`** — FastAPI entrypoint on port 8013. CORS for localhost
  (3000/5173/8013), `/health` endpoint, mounts `static/` for the FE,
  includes `ai_router`.
- **`ai_router.py`** — `APIRouter(prefix="/api/ai")` wrapping existing
  AI logic from `updated_universal_service_bot.UniversalServiceBot`.
  Lazy bot init (so the server boots even if ChromaDB isn't seeded yet).
  Canonical 503 `{"error": "AI not configured"}` when
  `OPENROUTER_API_KEY` is missing on LLM-using endpoints.
- **`static/index.html`** — vanilla-JS console with one form per AI
  endpoint, dark theme, status pill that hits `/api/ai/status`. No
  build step required (served by FastAPI's `StaticFiles`).
- **`start.sh`** — `uvicorn app:app --reload --port 8013`.

### Endpoints exposed under `/api/ai/*`
| Method | Path | Wraps |
|---|---|---|
| POST | `/api/ai/chat` | `UniversalServiceBot.chatAway` |
| POST | `/api/ai/chat-stateless` | `UniversalServiceBot.chatAway2` |
| POST | `/api/ai/detect-sector` | `UniversalServiceBot.detect_sector_with_ai` |
| POST | `/api/ai/detect-sector-keywords` | `UniversalServiceBot.detect_with_keywords` |
| POST | `/api/ai/ingredients` | `UniversalServiceBot.getIngredients` (+ `load_sector_prompt` fallback) |
| POST | `/api/ai/process-conversation` | `UniversalServiceBot.process_conversation` |
| GET  | `/api/ai/sectors` | `UniversalServiceBot.get_available_sectors` |
| GET  | `/api/ai/sectors/{sector}` | `UniversalServiceBot.get_sector_info` |
| GET  | `/api/ai/status` | runtime probe (key set? bot initialized? init error?) |

`/health` is also exposed at the app root.

### Why `OrderProcessor` was not wrapped
`orderProcessor.py` imports `from menuIndexer import MenuIndexer, MenuParser`
but only `menuIndexerIntentCalc.py` exists in the repo — i.e. the import is
already broken at module load. Wrapping it would crash the server. Out of
scope here; flagged as a pre-existing repo bug.

### Verification
- `python3 -m py_compile app.py ai_router.py` → both pass.
- No `pip install` was run (per scope).
- FastAPI + uvicorn already listed in `requirements.txt` (lines 2-3),
  so no requirements changes were necessary.

### To launch
```bash
pip install -r requirements.txt          # one-time
export OPENROUTER_API_KEY=sk-or-...      # optional; w/o it /api/ai/chat returns 503
./start.sh
# → open http://localhost:8013/
```

Note: This new `app.py` runs alongside the pre-existing `admin_app.py`
(port-configurable Admin Dashboard) and `amazon_main.py` /
`eleven_labs_main.py` (Twilio voice/SMS bridges). They are separate
processes serving different surfaces.

### Files touched
- Added: `app.py`, `ai_router.py`, `static/index.html`, `start.sh`.
- Modified: `_AUDIT_NOTE.md` (this section).
- No changes to the existing AI scripts (functions are wrapped, not rewritten).
- Log: `/Users/erolakarsu/projects/_AUDIT/apply3_logs/groupB_GenericOrderingService.md`.
