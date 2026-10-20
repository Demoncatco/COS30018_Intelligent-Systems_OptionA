# Model Service: the HTTP tool the Diagnostic Agent calls to score one MRI study.
# Every mode returns the same PredictionResponse contract from shared/schema.py, so agents never need to know which mode is running.
#
# MODEL_MODE (set in .env) picks where the scores come from:
#   stub   = random but repeatable scores, seeded by study_id. model_version "stub-0.1.0". For building and testing agents.
#   replay = precomputed scores from predictions.json, exported on Kaggle by model_service/knee_model/export_predictions.py.
#            Real model output with no GPU and no dataset needed. Unknown study_ids get a 404.
#   live   = runs the trained checkpoint on DICOMs downloaded to LIVE_DATASET_ROOT. Needs torch and the data.
#
# Run the web server via --> uvicorn model_service.api.main:app --port 8001 --reload
# Once the web server is running, http://localhost:8001/docs provides auto-generated documentation.
# Port 8001 matches MODEL_SERVICE_URL in .env; port 8000 is kept for the backend.

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Literal, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from shared.config import Settings, get_settings
from shared.schema import (
    FINDINGS,
    PLANES,
    PlaneScores,
    PredictionRequest,
    PredictionResponse,
    ReplayPredictionsFile,
)

logger = logging.getLogger(__name__)

MODEL_VERSION = "stub-0.1.0"  # Version reported by stub mode. Replay and live report the checkpoint's version instead.

ModelMode = Literal["stub", "replay", "live"]


class StudyNotFoundError(KeyError):
    """The backend has no data for this study_id."""


class Backend(Protocol):
    mode: ModelMode
    model_version: str

    def predict(self, study_id: str) -> PredictionResponse: ...

    def study_ids(self) -> list[str] | None: ...


# ------------------------------------------------------------
# stub: random but deterministic per study_id
# ------------------------------------------------------------
class StubBackend:
    mode: ModelMode = "stub"
    model_version = MODEL_VERSION

    def predict(self, study_id: str) -> PredictionResponse:
        start = time.perf_counter()  # Counter that will be used for latency_ms.

        # A private random number generator seeded by study_id, so the same study_id always gets the same numbers.
        rng = random.Random(study_id)

        # One score per finding per plane, generated to reflect what the actual model will show.
        per_plane = [
            PlaneScores(plane=plane, scores={name: rng.random() for name in FINDINGS})
            for plane in PLANES
        ]

        # The headline score for each finding is the mean over the three planes, so findings == mean(per_plane) holds.
        findings = {
            name: sum(ps.scores[name] for ps in per_plane) / len(per_plane)
            for name in FINDINGS
        }

        # Random knee side. "unknown" always comes with "unavailable", matching what the real model can report.
        laterality = rng.choice(["left", "right", "unknown"])
        laterality_source = "unavailable" if laterality == "unknown" else "dicom_tag"

        # Flags low_slice_count roughly 1 in 5 times so the agents exercise warning paths, not just the happy path.
        quality_flags = ["low_slice_count"] if rng.random() < 0.2 else []

        return PredictionResponse(
            study_id=study_id,
            findings=findings,
            per_plane=per_plane,
            laterality=laterality,
            laterality_source=laterality_source,
            quality_flags=quality_flags,
            model_version=self.model_version,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def study_ids(self) -> list[str] | None:
        return None  # any study_id is accepted


# ------------------------------------------------------------
# replay: precomputed predictions exported from Kaggle
# ------------------------------------------------------------
class ReplayBackend:
    mode: ModelMode = "replay"

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"replay file not found: {path}. Export it on Kaggle with "
                "model_service.knee_model.export_predictions, or set MODEL_MODE=stub."
            )
        replay = ReplayPredictionsFile.model_validate_json(path.read_text(encoding="utf-8"))
        self.model_version = replay.model_version
        self.source = replay.source
        self._predictions = {p.study_id: p for p in replay.predictions}
        logger.info("replay mode: %d studies from %s (%s)", len(self._predictions), path, self.model_version)

    def predict(self, study_id: str) -> PredictionResponse:
        start = time.perf_counter()
        stored = self._predictions.get(study_id)
        if stored is None:
            raise StudyNotFoundError(study_id)
        # latency_ms is this service's own time, as the contract says; the Kaggle inference time is not replayed.
        return stored.model_copy(update={"latency_ms": (time.perf_counter() - start) * 1000}, deep=True)

    def study_ids(self) -> list[str] | None:
        return sorted(self._predictions)


# ------------------------------------------------------------
# live: the trained checkpoint on local DICOMs
# ------------------------------------------------------------
class LiveBackend:
    mode: ModelMode = "live"

    def __init__(self, settings: Settings) -> None:
        import pandas as pd

        from model_service.knee_model.inference import KneePredictor

        self.dataset_root = Path(settings.live_dataset_root)
        self.split_folder = settings.live_split_folder
        series_csv = Path(settings.live_series_csv)
        if not series_csv.exists():
            raise FileNotFoundError(f"series table not found: {series_csv} (set LIVE_SERIES_CSV)")
        if not (self.dataset_root / self.split_folder).is_dir():
            raise FileNotFoundError(f"DICOM folder not found: {self.dataset_root / self.split_folder} (set LIVE_DATASET_ROOT)")
        self.series_table = pd.read_csv(series_csv)
        self.predictor = KneePredictor(settings.model_checkpoint, device=settings.model_device)
        self.model_version = self.predictor.model_version
        logger.info("live mode: %s on %s", self.model_version, self.predictor.device)

    def predict(self, study_id: str) -> PredictionResponse:
        if not (self.series_table["StudyInstanceUID"] == study_id).any():
            raise StudyNotFoundError(study_id)
        study_dir = self.dataset_root / self.split_folder / study_id
        if not study_dir.is_dir():
            raise StudyNotFoundError(study_id)
        return self.predictor.predict_study(study_id, self.series_table, self.dataset_root, self.split_folder)

    def study_ids(self) -> list[str] | None:
        on_disk = {p.name for p in (self.dataset_root / self.split_folder).iterdir() if p.is_dir()}
        return sorted(on_disk & set(self.series_table["StudyInstanceUID"].astype(str)))


def build_backend(settings: Settings) -> Backend:
    if settings.model_mode == "stub":
        return StubBackend()
    if settings.model_mode == "replay":
        return ReplayBackend(settings.replay_predictions)
    if settings.model_mode == "live":
        return LiveBackend(settings)
    raise ValueError(f"unknown MODEL_MODE {settings.model_mode!r}")


# ------------------------------------------------------------
# API
# ------------------------------------------------------------
class HealthResponse(BaseModel):
    # Health Response Endpoint - whether the service is up, which mode it is in and which model it is serving.
    status: Literal["ok", "error"] = Field(description="'ok' when the backend loaded; 'error' otherwise (HTTP 503).")
    mode: ModelMode = Field(description="Value of MODEL_MODE.")
    model_version: str | None = Field(description="Version of the loaded model. 'stub-*' means random output.")
    n_studies: int | None = Field(default=None, description="Studies available in replay/live mode; None in stub mode.")
    detail: str = Field(default="", description="Why loading failed, when status is 'error'.")


class StudiesResponse(BaseModel):
    mode: ModelMode
    any_study_id_accepted: bool = Field(description="True in stub mode, where every study_id gets scores.")
    study_ids: list[str] = Field(default_factory=list)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    mode: ModelMode = settings.model_mode

    # A backend that fails to load (missing replay file, missing checkpoint) does not crash the import:
    # /health reports the reason and /predict returns 503, so the agents can show a clear error.
    backend: Backend | None
    load_error = ""
    try:
        backend = build_backend(settings)
    except Exception as error:  # noqa: BLE001
        backend = None
        load_error = f"{type(error).__name__}: {error}"
        logger.error("model service failed to load in %s mode: %s", mode, load_error)

    app = FastAPI(
        title="Knee Abnormality Scanner",
        description=f"Scores an MRI knee study for 12 findings. MODEL_MODE={mode}.",
        version=backend.model_version if backend else "unavailable",
    )
    app.state.backend = backend
    app.state.load_error = load_error

    def _require_backend() -> Backend:
        if backend is None:
            raise HTTPException(status_code=503, detail=f"model service not ready ({mode} mode): {load_error}")
        return backend

    @app.get("/health", response_model=HealthResponse, summary="Liveness check")
    def health() -> HealthResponse | JSONResponse:
        if backend is None:
            body = HealthResponse(status="error", mode=mode, model_version=None, detail=load_error)
            return JSONResponse(status_code=503, content=body.model_dump())
        ids = backend.study_ids()
        return HealthResponse(status="ok", mode=mode, model_version=backend.model_version,
                              n_studies=None if ids is None else len(ids))

    @app.get("/studies", response_model=StudiesResponse, summary="Study IDs this mode can score")
    def studies() -> StudiesResponse:
        ids = _require_backend().study_ids()
        return StudiesResponse(mode=mode, any_study_id_accepted=ids is None, study_ids=ids or [])

    @app.post("/predict", response_model=PredictionResponse, summary="Score one study")
    def predict(request: PredictionRequest) -> PredictionResponse:
        active = _require_backend()
        try:
            return active.predict(request.study_id)
        except StudyNotFoundError:
            raise HTTPException(status_code=404, detail=f"study {request.study_id} is not available in {mode} mode") from None
        except Exception as error:
            logger.exception("prediction failed for %s", request.study_id)
            raise HTTPException(status_code=500, detail=f"prediction failed: {type(error).__name__}: {error}") from error

    return app


# Module-level app for uvicorn. Built from .env, so MODEL_MODE there decides the mode.
app = create_app()