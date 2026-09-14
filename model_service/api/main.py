# Currently a Stub Model
# Placeholder for the knee-abnormality model. Provides the same response shape that the real model will use, which can be found in shared/schema.py
# model_version = stub-0.1.0 as this is currently a stub model, providing no real prediction until the model has been trained.
# Random probabilities are seeded by study_id so that they provide the same probabilities on return. Can be used as a repeatable test case for retry and escalation logic

# Run the web server via --> uvicorn model_service.api.main:app --reload
# Once web server is running, http://localhost:8000/docs will provide auto-generated documentation

import random
import time
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Imports the anomaly findings, planes, and formats for the request and response from the schema.py file
from shared.schema import (
    FINDINGS,
    PLANES,
    PlaneScores,
    PredictionRequest,
    PredictionResponse,
)

MODEL_VERSION = "stub-0.1.0"

# Creates the app object. Provides the Title, Description and Model Version
app = FastAPI(
    title="Knee Abnormality Scanner",
    description=(
        "Scores an MRI knee study for 12 findings."
    ),
    version=MODEL_VERSION,
)


class HealthResponse(BaseModel):
    # Health Response Endpoint - Defines the shape of the answer to whether the service is up and running

    status: Literal["ok"] = Field(description="Always 'ok' when the service is up.") # Status must be exactly the string "ok"
    model_version: str = Field(description="Version of the loaded model.")


# Runs when a GET request to /health has been made. 
@app.get("/health", response_model=HealthResponse, summary="Liveness check")
def health() -> HealthResponse:
    # Confirms that the service is running and also reports the model that has loaded.
    return HealthResponse(status="ok", model_version=MODEL_VERSION)

# Same idea as the above, but with a post request. Post used when caller agent sends data in body of the request. Called Agent receives study ID and full prediction back
@app.post("/predict", response_model=PredictionResponse, summary="Score one study")
def predict(request: PredictionRequest) -> PredictionResponse:

    # Return per-finding probability for one study. As part of the Stub Model, numbers are random but deterministic per study_id.
    start = time.perf_counter() # Counter that will be used for latency_ms.

    # A private random number generator which is used for study_id. Study_id seeded so that the same study_id will have the same random generated string.
    rng = random.Random(request.study_id)

    # One score per finding per plane, this is generated for the stub model, to reflect what the actual model will show.
    per_plane = [
        PlaneScores(plane=plane, scores={name: rng.random() for name in FINDINGS})
        for plane in PLANES
    ]

    # Calculates the findings score for each abnormality as the average for the three planes (sagittal, coronal and axial)
    # Adds up for example the score of ACL across all planes, then devides by the len(per_plane) which gives the mean score.
    # Keeps the stub fake data internally consistent so that agent's during testing can check that findings == per_plane
    findings = {
        name: sum(ps.scores[name] for ps in per_plane) / len(per_plane)
        for name in FINDINGS
    }

    # Picks a random knee side from the three allowed values
    laterality = rng.choice(["left", "right", "unknown"])
    # If the side is unknown then the source is unavailable (represents how the actual model will work), otherwise the stub model pretends the side (left or right) was read from a reliable DICOM tag.
    # This keeps logical consistency with what the actual model will bring out.
    laterality_source = "unavailable" if laterality == "unknown" else "dicom_tag"

    # Flags low_slice_count roughly every 1 in 5 times so that the agents can handle warning paths, not just the correct working path.
    quality_flags = ["low_slice_count"] if rng.random() < 0.2 else []


    latency_ms = (time.perf_counter() - start) * 1000 # per-counter returns seconds, multiplying by 1000 converts to ms. latency_ms required for PredictionResponse()

    # Formats the above sample response data into one collated PredictionResponse which can be sent back to the agent that called for it.
    return PredictionResponse(
        study_id=request.study_id,
        findings=findings,
        per_plane=per_plane,
        laterality=laterality,
        laterality_source=laterality_source,
        quality_flags=quality_flags,
        model_version=MODEL_VERSION,
        latency_ms=latency_ms,
    )
