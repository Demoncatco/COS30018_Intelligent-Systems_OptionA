"""FastAPI backend the frontend talks to: start sessions, read results, stream the live agent trace, run the baseline.

Run: uvicorn backend.main:app --port 8000   (docs at http://localhost:8000/docs)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.models import (
    EventsPage,
    SessionCreate,
    SessionDetail,
    SessionKind,
    SessionSummary,
)
from backend.sessions import QueueFullError, Services, SessionEntry, SessionManager
from backend.uploads import UploadStore
from model_service.knee_model.upload import UploadError, UploadSummary, is_upload_id
from shared.config import Settings, get_settings
from shared.logging_setup import setup_logging
from shared.schema import FINDINGS, PLANES, FinalReport

logger = logging.getLogger(__name__)

MAX_WAIT_S = 900.0  # longest a client may block on ?wait_s=
STREAM_POLL_S = 0.25  # how often the stream checks for new events
STREAM_HEARTBEAT_S = 15.0  # keeps proxies from closing an idle stream
REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # client-supplied ids are only echoed if they look safe
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"  # built React app, served at / when present


def create_app(settings: Settings | None = None, services: Services | None = None,
               configure_logging: bool = True) -> FastAPI:
    settings = settings or (services.settings if services else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if configure_logging:
            setup_logging()
            logger.info("backend settings", extra={"data": settings.safe_summary()})
        active = services or Services.build(settings)
        app.state.services = active
        app.state.manager = SessionManager(active)
        app.state.uploads = UploadStore(active.settings)
        logger.info("agents wired", extra={"data": {"notes": active.runners.notes}})
        yield
        app.state.manager.shutdown()
        if services is None:  # only close what we built
            active.close()

    app = FastAPI(title="Knee MRI Multi-Agent Backend", version="1.0.0", lifespan=lifespan,
                  description="Sessions, live agent trace and the single-call baseline.")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
                       allow_headers=["*"], expose_headers=["Location", "X-Request-ID"])

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        # One access-log line per request, tagged with an X-Request-ID the client can quote in bug reports.
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if REQUEST_ID.match(supplied) else uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled error", extra={"data": {"request_id": request_id, "path": request.url.path}})
            response = JSONResponse(status_code=500, content={"detail": "internal server error",
                                                              "request_id": request_id})
        response.headers["X-Request-ID"] = request_id
        logger.info("%s %s %s", request.method, request.url.path, response.status_code,
                    extra={"data": {"request_id": request_id, "status": response.status_code,
                                    "ms": round((time.perf_counter() - started) * 1000, 1)}})
        return response

    def manager(request: Request) -> SessionManager:
        return request.app.state.manager

    def require(request: Request, session_id: str) -> SessionEntry:
        entry = manager(request).get(session_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"session {session_id} not found")
        return entry

    # --- Health ---

    @app.get("/health", summary="Model Service, Ollama and agent wiring")
    def health(request: Request, deep: bool = Query(True, description="false = skip the Ollama/Model Service checks")):
        services: Services = request.app.state.services
        runners = services.runners
        body: dict = {
            "agents": {"diagnostic": runners.diagnostic_impl, "recommendation": runners.recommendation_impl,
                       "notes": runners.notes},
            "sessions": manager(request).counts(),
            "config": {"model_mode": settings.model_mode, "report_threshold": settings.report_threshold,
                       "low_confidence_threshold": settings.low_confidence_threshold,
                       "max_recheck_rounds": settings.max_recheck_rounds,
                       "models": {a: settings.model_for(a) for a in ("diagnostic", "coordinator",
                                                                    "recommendation", "baseline")}},
        }
        if deep:
            body["model_service"] = services.model_service.health()
            body["ollama"] = services.llm.health()
            ready = (body["model_service"].get("status") == "ok" and body["ollama"].get("ready") is True
                     and runners.recommendation_impl == "agent")
            body["status"] = "ok" if ready else "degraded"
        else:
            body["status"] = "ok"
        return body

    @app.get("/studies", summary="Study IDs the Model Service can score (proxied)")
    def studies(request: Request):
        try:
            return request.app.state.services.model_service.studies()
        except Exception as error:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Model Service: {type(error).__name__}: {error}") from None

    # --- Uploads ---

    def uploads(request: Request) -> UploadStore:
        return request.app.state.uploads

    @app.post("/uploads", response_model=UploadSummary, status_code=201, summary="Upload one knee MRI study")
    def create_upload(request: Request, files: Annotated[list[UploadFile], File(description=".dcm files or a .zip")]):
        # Sync endpoint: FastAPI runs it in a worker thread, so unzipping and preprocessing do not block the stream.
        try:
            return uploads(request).ingest(files)
        except UploadError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None

    @app.get("/uploads/{upload_id}", response_model=UploadSummary, summary="What was found in an upload")
    def get_upload(request: Request, upload_id: str):
        return uploads(request).summary(upload_id)

    @app.get("/uploads/{upload_id}/slices/{plane}/{index}.png", summary="One preprocessed slice, as the model sees it")
    def get_slice(request: Request, upload_id: str, plane: str, index: int):
        if plane not in PLANES:
            raise HTTPException(status_code=404, detail=f"unknown plane {plane!r}")
        png = uploads(request).slice_png(upload_id, plane, index)
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "max-age=86400, immutable"})

    @app.delete("/uploads/{upload_id}", status_code=204, summary="Delete an uploaded study")
    def delete_upload(request: Request, upload_id: str):
        uploads(request).delete(upload_id)
        return Response(status_code=204)

    # --- Sessions ---

    def start(request: Request, response: Response, body: SessionCreate, kind: SessionKind,
              wait_s: float) -> SessionDetail:
        if is_upload_id(body.study_id) and not request.app.state.uploads.exists(body.study_id):
            raise HTTPException(status_code=404, detail=f"upload {body.study_id} not found; upload the scan again")
        try:
            entry = manager(request).start(body.study_id, body.patient_metadata, kind=kind, driver=body.driver)
        except QueueFullError as error:
            raise HTTPException(status_code=429, detail=str(error), headers={"Retry-After": "30"}) from None
        finished = manager(request).wait(entry, min(wait_s, MAX_WAIT_S))
        response.status_code = 200 if finished else 202
        response.headers["Location"] = f"/sessions/{entry.session_id}"
        return detail(entry)

    @app.post("/sessions", response_model=SessionDetail, status_code=202, summary="Start a multi-agent session")
    def create_session(request: Request, response: Response, body: SessionCreate,
                       wait_s: float = Query(0, ge=0, description="Block up to this long for the result")):
        return start(request, response, body, "multi_agent", wait_s)

    @app.post("/baseline", response_model=SessionDetail, status_code=202, summary="Start a single-call baseline")
    def create_baseline(request: Request, response: Response, body: SessionCreate,
                        wait_s: float = Query(0, ge=0, description="Block up to this long for the result")):
        return start(request, response, body, "baseline", wait_s)

    @app.get("/sessions", response_model=list[SessionSummary], summary="Recent sessions, newest first")
    def list_sessions(request: Request, limit: int = Query(50, ge=1, le=500)):
        return [summary(e) for e in manager(request).list(limit)]

    @app.get("/sessions/{session_id}", response_model=SessionDetail, summary="Status, final report and usage")
    def get_session(request: Request, session_id: str, include_events: bool = False):
        return detail(require(request, session_id), include_events)

    @app.get("/sessions/{session_id}/report", response_model=FinalReport, summary="Final report once finished")
    def get_report(request: Request, session_id: str):
        entry = require(request, session_id)
        if not entry.done.is_set() or entry.state.final_report is None:
            raise HTTPException(status_code=409, detail=f"session is {entry.state.status.value}; no report yet")
        return entry.state.final_report

    @app.get("/sessions/{session_id}/events", response_model=EventsPage, summary="Trace events (polling)")
    def get_events(request: Request, session_id: str, after: int = Query(-1, ge=-1)):
        entry = require(request, session_id)
        finished = entry.done.is_set()  # read before the events so none are missed
        events = list(entry.state.events)[after + 1:]
        return EventsPage(session_id=session_id, status=entry.state.status, finished=finished, events=events,
                          next_after=events[-1].seq if events else after)

    @app.get("/sessions/{session_id}/stream", summary="Live trace as Server-Sent Events")
    async def stream(request: Request, session_id: str, after: int = Query(-1, ge=-1),
                     last_event_id: str | None = Header(None)):
        entry = require(request, session_id)
        if last_event_id and last_event_id.isdigit():  # browser reconnect
            after = max(after, int(last_event_id))
        return StreamingResponse(_event_stream(request, entry, after), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if FRONTEND_DIST.joinpath("index.html").exists():  # after the API routes, so they take priority
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

    return app


async def _event_stream(request: Request, entry: SessionEntry, after: int) -> AsyncIterator[str]:
    # "agent_event" per AgentEvent (id = seq), then one "end" with the final status.
    cursor, idle = after + 1, 0.0
    while True:
        finished = entry.done.is_set()
        events = list(entry.state.events)[cursor:]
        for event in events:
            yield f"id: {event.seq}\nevent: agent_event\ndata: {event.model_dump_json()}\n\n"
        cursor += len(events)
        if finished:
            end = {"session_id": entry.session_id, "status": entry.state.status.value}
            yield f"event: end\ndata: {json.dumps(end)}\n\n"
            return
        if await request.is_disconnected():
            return
        idle = 0.0 if events else idle + STREAM_POLL_S
        if idle >= STREAM_HEARTBEAT_S:
            idle = 0.0
            yield ": keep-alive\n\n"
        await asyncio.sleep(STREAM_POLL_S)


def summary(entry: SessionEntry) -> SessionSummary:
    state, sid = entry.state, entry.session_id
    return SessionSummary(
        session_id=sid, study_id=state.study_id, produced_by=entry.kind, driver=entry.driver, status=state.status,
        finished=entry.done.is_set(), created_at=state.created_at, updated_at=state.updated_at,
        n_events=len(state.events),
        links={"self": f"/sessions/{sid}", "events": f"/sessions/{sid}/events",
               "stream": f"/sessions/{sid}/stream", "report": f"/sessions/{sid}/report"},
    )


def detail(entry: SessionEntry, include_events: bool = False) -> SessionDetail:
    state = entry.state
    usage = {agent: dict(totals) for agent, totals in list(entry.usage.items())}  # snapshot; workers may still add
    if usage:
        usage["total"] = {k: round(sum(u[k] for u in usage.values()), 1)
                          for k in ("calls", "tokens_in", "tokens_out", "latency_ms")}
    return SessionDetail(
        **summary(entry).model_dump(), patient_metadata=state.patient_metadata,
        final_report=state.final_report if entry.done.is_set() else None, errors=list(state.errors),
        disagreements=list(state.disagreements), recheck_rounds=state.recheck_rounds, user_action=state.user_action,
        usage=usage, stats=entry.stats, events=list(state.events) if include_events else None,
        scores=_latest_scores(entry),
    )


def _latest_scores(entry: SessionEntry) -> dict[str, float] | None:
    # The Diagnostic Agent's last result, or (baseline) the Model Service scores in the trace.
    if entry.state.latest_diagnostic is not None:
        return dict(entry.state.latest_diagnostic.findings)
    for event in reversed(list(entry.state.events)):
        found = event.data.get("findings")
        if isinstance(found, dict) and set(found) == set(FINDINGS):
            return {name: float(found[name]) for name in FINDINGS}
    return None


app = create_app()