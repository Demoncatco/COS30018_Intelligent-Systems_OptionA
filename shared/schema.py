# The contract that will be used between the detection model and the other agents such as the diagnostic and recommendation agennts

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The findings are the 12 anomalies listed exactly as they are in the train.csv columns.
FINDINGS: list[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# The three MRI imaging planes present in every study.
# Sagittal = Side On Slice
# Coronal = Front On Slice
# Axial = Top Down Slice
Plane = Literal["sagittal", "coronal", "axial"] # Literal means that it must be one of these three exact values. This is a rule for what Plane could be equal to
PLANES: tuple[Plane, ...] = ("sagittal", "coronal", "axial") 

Laterality = Literal["left", "right", "unknown"] # Which Knee. This is required because Medial (inside of knee) is in different locations for a left knee or a right knee.
LateralitySource = Literal["dicom_tag", "inferred", "unavailable"] # Where the Knee answer came from. 
# Gives us three probability groups for the Laterality source:
# dicom_tag = reliable
# inferred = guessed from image, not as reliable
# unavailable = not sure


# Function receives a group of scores for each anomaly. The scores may look like {"ACL": 0.8, "MCL": 0.1,} etc.
# Function does three checks on the scores.
def _check_scores(scores: dict[str, float]) -> dict[str, float]:
    # Checks that every one of the 12 anomalies are present, there are no extra columns that have been found.
    expected = set(FINDINGS) # A set is used here because order does not matter and duplicates are dropped. Allows us to compare the 12 anomalies without worrying about order
    got = set(scores) # Provides the set of anomalies that were actually sent through
    if got != expected: # If the two sets are not identical, then something is extra or missing which is incorrect
        missing = sorted(expected - got) # Lets us know the anomaly names that we wanted but did not get
        extra = sorted(got - expected) # Lets us know additional anomaly names that we go but did not want.

        # Stops function and provides and error and output on what may have been missing, or unexpected.
        raise ValueError(
            f"scores must contain exactly the {len(FINDINGS)} known findings; "
            f"missing={missing} unexpected={extra}"
        )

    # For each name that was given, if the probability 'p' is not between 0 and 1 (0.0<=p<=1.0) then error is thrown.
    # If there is a value that is outside of this range, assign variable bad to that so the error can be thrown.
    bad = {name: p for name, p in scores.items() if not 0.0 <= p <= 1.0}
    if bad:
        raise ValueError(f"probabilities must be between 0 and 1, got {bad}")
    return scores


class PlaneScores(BaseModel):

# Must be a Plane and holds the value of what MRI plane (sagittal, coronal or axial) that the scores were computed from
    plane: Plane = Field(description="Which MRI plane these scores were computed from.")
    scores: dict[str, float] = Field(
        description=(
            "Probability ranging from 0 - 1 of each anomaly finding, keyed by the finding name, using only the slices from this plane. Always contains all 12 findings"
        )
    )
    
    # Runs the check scores function that checks if all keys are exactly the 12 anomaly findings, if every value is between 0 and 1.
    _validate_scores = field_validator("scores")(_check_scores)

# This is the body that agents will send to Post /predict to find out about the scan.
# PredictionRequest is the shape of the question
class PredictionRequest(BaseModel):

    study_id: str = Field(
        description = "Identifier of the MRI study to score, as it appears in the dataset.", 
        min_length = 1, # Rejects an empty string
        examples = ["1.2.826.0.1.3680043.8.498.10001"], # Example of a DICOM study Identifer
    )
    # When agent send PredictionRequest, the message sent will look like {"study_id" : "12345"}

# This is the body that the agent will receive from the scan score
# PredictionResponse is the shape of the answer
class PredictionResponse(BaseModel):
    # Everything that the model finds from the one study (group of slices from different planes)
    # 'findings' is the header for each finding, providing overall summary for each 12 anomalies, 'per_plane' is the findings per plane (sagittal, coronal and axial)
    # Breaking down per plane will allow the diagnostic agent to reason between different planes as the score may return ACL tear at 0.9 for Sagittal, but 0.2 for coronal.
    study_id: str = Field(description="Echo of the requested study identifier.")

    # Dictionary has key and value pairs. The keys are the anomaly findings and the values are the probability numbers.
    # This will look something like the below:
    # {
    #  "ACL": 0.87,
    #  "MCL": 0.12,
    #  ...
    # }
    findings: dict[str, float] = Field(
        description=(
            "Overall probability (0-1) of each finding being present, keyed by finding name. Always contains all 12 anomalies."
        )
    )

    # This will be the scores for every anomaly per plane:
    # This will look something like the below:
    # {"plane": "sagittal", "scores": {"ACL": 0.93, ...}},
    # {"plane": "coronal", "scores": {"ACL": 0.81, ...}},
    # {"plane": "axial", "scores": {"ACL": 0.97, ...}}
    per_plane: list[PlaneScores] = Field(
        description=(
            "The same probabilities computed separately for each imaging plane. One entry per available plane (normally sagittal, coronal and axial)."
        )
    )
    laterality: Laterality = Field(description="Which knee was imaged.")
    laterality_source: LateralitySource = Field(
        description=(
            "Where the laterality came from, three types: read from a DICOM tag (classified as reliable), inferred from the images (classified as less reliable), or unavailable."
        )
    )
    quality_flags: list[str] = Field(
        default_factory=list, # Makes a new list if the agent sender leaves it out.
        description=(
            "Warnings about the input that may reduce confidence, e.g. 'low_slice_count' or 'missing_plane:axial'. Empty when nothing is flagged."
        ),
    )
    model_version: str = Field(
        description="Version string of the model that produced this response. 'stub-*' means random output."
    )
    latency_ms: float = Field(
        ge=0, # greater than or equal to 0, Pydantic rejects negatives.
        description="Wall-clock time the service spent producing this response, in milliseconds.",
    )

    # Runs the check scores function that checks if all keys are exactly the 12 anomaly findings, if every value is between 0 and 1.
    _validate_findings = field_validator("findings")(_check_scores)

    # This applies any settings established in it to the whole form that is sent back to the agent that requested it
    model_config = ConfigDict(
        json_schema_extra={
            # Example below is how a response would be formatted
            "examples": [
                {
                    "study_id": "1.2.826.0.1.3680043.8.498.10001",
                    "findings": {name: 0.1 for name in FINDINGS}
                    | {"ACL": 0.87, "Effusion": 0.64}, # | merges two dictionaries together
                    "per_plane": [
                        {
                            "plane": "sagittal",
                            "scores": {name: 0.1 for name in FINDINGS} | {"ACL": 0.93},
                        },
                        {
                            "plane": "coronal",
                            "scores": {name: 0.1 for name in FINDINGS} | {"ACL": 0.81},
                        },
                        {
                            "plane": "axial",
                            "scores": {name: 0.1 for name in FINDINGS}
                            | {"Effusion": 0.7},
                        },
                    ],
                    "laterality": "left",
                    "laterality_source": "dicom_tag",
                    "quality_flags": [],
                    "model_version": "stub-0.1.0",
                    "latency_ms": 12.4,
                }
            ]
        } # pyright: ignore[reportArgumentType]
    )


# ============================================================
# Coordinator Agent <-> Diagnostic Agent contract
# ============================================================
class DiagnosticDecision(str, Enum):
    FINALIZE = "finalize"   
    RETRY = "retry" 
    ESCALATE = "escalate" 

def _check_partial_scores(scores: dict[str, float]) -> dict[str, float]:
    unknown = sorted(set(scores) - set(FINDINGS))
    if unknown:
        raise ValueError(f"unexpected finding names: {unknown}")

    bad = {name: p for name, p in scores.items() if not 0.0 <= p <= 1.0}
    if bad:
        raise ValueError(f"probabilities must be between 0 and 1, got {bad}")
    return scores

class DiagnosticAgentRequest(BaseModel):
    study_id: str = Field(
        min_length=1,
        examples=["1.2.826.0.1.3680043.8.498.10001"],
    )

    task_type: str = Field(
        default="initial",
        description=(
            'Either "initial" or "recheck: <specific ask>'
        ),
    )

    focus_findings: list[str] = Field(
        default_factory=list,
        description=(
            "Findsing to re-examine during recheck."
        ),
    )

    @field_validator("focus_findings")
    @classmethod
    def _validate_focus_findings(cls, names: list[str]) -> list[str]:
        unknown = sorted(set(names) - set(FINDINGS))
        if unknown:
            raise ValueError(f"focus_findings contains unknown finding names: {unknown}")
        return names

class DiagnosticAgentResponse(BaseModel):
    study_id: str
    findings: dict[str, float] = Field(
        description=(
            "Final probability for each finding."
        )
    )

    per_finding_confidence: dict[str, float] = Field(
        description=(
            "Confidence in each finding probability."
        )
    )

    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Overall confidence for the study."
        ),
    )

    decision: DiagnosticDecision = Field(
        description="Result of the diagnostic decision loop.",
    )

    retry_count: int = Field(
        ge=0,
        description="Numbers of Model Service retries.",
    )

    cross_plane_variance: dict[str, float] = Field(
        default_factory=dict,
        description="Findings with significant variation between imaging planes.",
    )

    quality_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Quality issues identified during processing."
        ),
    )

    laterality: Laterality = Field( description="Knee laterality.", )

    impression: str = Field(
        default="",
        description=(
            "Short plain-language initial impression from the Diagnostic Agent, e.g. 'Likely ACL tear with "
            "joint effusion; meniscus findings uncertain'. Passed on to the Recommendation Agent."
        ),
    )
    
    trace: list[str] = Field(
        default_factory=list,
        description="Diagnostic decision loop trace.",
    )

    _validate_findings = field_validator("findings")(_check_scores)
    _validate_per_finding_confidence = field_validator("per_finding_confidence")(_check_scores)
    _validate_cross_plane_variance = field_validator("cross_plane_variance")(_check_partial_scores)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "study_id": "1.2.826.0.1.3680043.8.498.10001",
                    "findings": {name: 0.05 for name in FINDINGS} | {"ACL": 0.88},
                    "per_finding_confidence": {name: 0.9 for name in FINDINGS} | {"ACL": 0.95},
                    "overall_confidence": 0.9,
                    "decision": "finalize",
                    "retry_count": 0,
                    "cross_plane_variance": {},
                    "quality_flags": [],
                    "laterality": "left",
                    "trace": [
                        "QC passed: all 3 planes present, slice counts nominal.",
                        "Called Model Service /predict.",
                        "ACL 0.88 consistent across all planes (variance below threshold).",
                        "Decision: finalize.",
                    ],
                }
            ]
        } 
    )

# ============================================================
# Coordinator Agent <-> Recommendation Agent contract
# ============================================================

class UrgencyLevel(str, Enum):
    ROUTINE = "routine"
    SEE_DOCTOR_SOON = "see_doctor_soon"
    URGENT = "urgent"


class EvidenceStatus(str, Enum):
    GROUNDED_LOCAL = "grounded_local"
    GROUNDED_WEB = "grounded_web"
    NO_CREDIBLE_SOURCE_FOUND = "no_credible_source_found"


class CoordinatorRequest(BaseModel):
    # Sent from the Coordinator Agent to the Recommendation Agent

    finding: dict[str, float] = Field(
        description=(
            "Probability (0-1) of each of the 12 findings, using the exact same "
            "names as FINDINGS above - e.g. 'Baker's', 'PF OA', 'Medial Meniscus'."
        )
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Overall confidence from the Diagnostic Agent."
    )
    study_context: dict = Field(
        default_factory=dict, description="Minimal relevant context only, e.g. {'laterality': 'left'}."
    )
    task_type: str = Field(
        description='Either "initial" or "recheck: <specific ask>" (set by the Coordinator retry loop).'
    )
    impression: str = Field(
        default="",
        description="The Diagnostic Agent's initial impression, passed through unchanged by the Coordinator.",
    )

    # Reuses the SAME validator
    _validate_finding = field_validator("finding")(_check_scores)


class RecommendationResponse(BaseModel):
    # Returned from the Recommendation Agent to the Coordinator Agent

    recommendation_text: str
    citations: list[str] = Field(default_factory=list)
    urgency: UrgencyLevel
    evidence_status: EvidenceStatus
    trace: list[str] = Field(default_factory=list, description="Step-by-step log, for the UI's agent trace.")
    concerns: list[str] = Field(
        default_factory=list,
        description=(
            "Clinical concerns about the Diagnostic Agent's findings, e.g. 'Fracture 0.8 without Contusion or "
            "Effusion is an unusual pattern'. Empty when the Recommendation Agent has no concerns."
        ),
    )
    suggested_recheck: list[str] = Field(
        default_factory=list,
        description=(
            "Finding names (exact FINDINGS spelling) the Recommendation Agent thinks the Diagnostic Agent "
            "should re-examine. The Coordinator decides whether a recheck actually happens."
        ),
    )

    @field_validator("suggested_recheck")
    @classmethod
    def _validate_suggested_recheck(cls, names: list[str]) -> list[str]:
        unknown = sorted(set(names) - set(FINDINGS))
        if unknown:
            raise ValueError(f"suggested_recheck contains unknown finding names: {unknown}")
        return names


# ============================================================
# Coordinator <-> UI contract
# ============================================================

class PatientMetadata(BaseModel):
    age: int | None = Field(default=None, ge=0, le=120)
    sex: Literal["female", "male", "other", "unknown"] = "unknown"
    symptoms: str = Field(default="", max_length=1000, description="Free text, e.g. 'pain after twisting, swelling'.")
    injury_mechanism: str = Field(default="", max_length=500)


class AgentName(str, Enum):
    COORDINATOR = "coordinator"
    DIAGNOSTIC = "diagnostic"
    RECOMMENDATION = "recommendation"
    MODEL_SERVICE = "model_service"
    BASELINE = "baseline"


class AgentEvent(BaseModel):
    # One line of the UI's live agent trace.
    session_id: str
    seq: int = Field(ge=0, description="Order of the event within the session, starting at 0.")
    timestamp: float = Field(description="Unix time in seconds.")
    agent: AgentName
    kind: Literal["decision", "tool_call", "tool_result", "message", "error", "status"]
    message: str
    data: dict = Field(default_factory=dict, description="Structured details, e.g. tool arguments or scores.")
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: float | None = Field(default=None, ge=0)


class SessionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"          # full report produced
    PARTIAL = "partial"              # report produced but an agent failed or evidence is missing
    INCONCLUSIVE = "inconclusive"    # images not good enough; user asked to re-upload
    FAILED = "failed"                # nothing useful could be produced


class ReportedFinding(BaseModel):
    name: str
    probability: float = Field(ge=0.0, le=1.0)
    per_plane: dict[str, float] = Field(default_factory=dict, description="Plane name -> probability.")
    low_confidence: bool = False

    @field_validator("name")
    @classmethod
    def _validate_name(cls, name: str) -> str:
        if name not in FINDINGS:
            raise ValueError(f"unknown finding name: {name}")
        return name


DISCLAIMER = (
    "This output is generated by an experimental AI system for educational purposes. It may indicate possible "
    "findings but is not a diagnosis. Please consult a qualified clinician about any knee symptoms or imaging."
)


class FinalReport(BaseModel):
    # Produced by both the multi-agent system and the baseline.
    session_id: str
    study_id: str
    status: SessionStatus
    produced_by: Literal["multi_agent", "baseline"] = "multi_agent"
    findings: list[ReportedFinding] = Field(default_factory=list, description="Findings above the reporting threshold.")
    impression: str = ""
    recommendation_text: str = ""
    citations: list[str] = Field(default_factory=list)
    urgency: UrgencyLevel | None = None
    evidence_status: EvidenceStatus | None = None
    overall_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    disagreements: list[str] = Field(default_factory=list, description="Unresolved Diagnostic/Recommendation disagreements.")
    errors: list[str] = Field(default_factory=list)
    user_action: str = Field(default="", description="What the user should do next, e.g. re-upload with an axial series.")
    coordination_summary: str = ""
    disclaimer: str = DISCLAIMER
    recheck_rounds: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    total_latency_ms: float = Field(default=0.0, ge=0)

    @field_validator("disclaimer")
    @classmethod
    def _disclaimer_cannot_be_removed(cls, text: str) -> str:
        return text if DISCLAIMER in text else f"{text}\n\n{DISCLAIMER}".strip()