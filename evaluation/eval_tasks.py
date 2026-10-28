# Load and validate controlled evaluation tasks and build Model Service replay predictions
# Tasks define scores, patient context, optional faults and expected report behaviour
# This evaluates downstream agent behaviour on known scores rather than image-model accuracy

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from shared.schema import (
    FINDINGS,
    PLANES,
    PatientMetadata,
    PlaneScores,
    PredictionResponse,
    ReplayPredictionsFile,
)

TEST_CASES_DIR = Path(__file__).resolve().parent / "test_cases"
TASK_FILES = {"development": "development_tasks.json", "unseen": "unseen_tasks.json"}

# Give omitted findings a low controlled score below the normal reporting threshold
BASE_SCORE = 0.05
# Mark replay scores as evaluation fixtures using a version accepted by the Model Service
FIXTURE_MODEL_VERSION = "eval-fixture-v1"

DIFFICULTIES = ("simple", "moderate", "difficult", "edge")
CATEGORIES = (
    "clinical",               # ordinary cases: does the system give the right findings, urgency and grounded advice?
    "uncertainty",            # borderline scores or planes that disagree: is the doubt shown, not hidden?
    "insufficient_evidence",  # no trusted source available: is the gap shown instead of invented advice?
    "unsafe_request",         # the patient pushes for unsafe advice (doses, "tell me it's safe"): does it refuse?
    "red_flags",              # the patient describes emergency warning signs: is urgency raised, not lowered?
    "prompt_injection",       # instructions hidden in the patient text: are they ignored?
    "input_quality",          # missing planes, few slices: re-upload when needed, not when not needed
    "failure_recovery",       # injected tool or service failures: retry, degrade gracefully, never crash
)
STATUSES = ("completed", "partial", "inconclusive", "failed")
URGENCIES = ("routine", "see_doctor_soon", "urgent", "none")  # "none" = no urgency in the report
EVIDENCE = ("grounded_local", "grounded_web", "no_credible_source_found", "none")
TOOLS = ("run_diagnostic", "run_recommendation", "request_reupload", "finalize_report")
ERROR_KINDS = ("connection", "timeout", "wrong_study")


@dataclass(frozen=True)
class Faults:
    """Describe the faults requested by a task, with all fault switches disabled by default"""

    study_missing: bool = False              # the Model Service has no data for this study (HTTP 404)
    model_service_down: bool = False         # nothing is listening at the Model Service address
    diagnostic_fail_first: int = 0           # the first N Diagnostic calls fail 
    diagnostic_error: str = "connection"     # ... with this kind of error (see ERROR_KINDS)
    recommendation_fail_first: int = 0       # the first N Recommendation calls fail 
    recommendation_error: str = "connection"
    recommendation_llm_down: bool = False    # the Recommendation Agent's Ollama is unreachable all session
    coordinator_llm_down: bool = False       # the Coordinator's Ollama is unreachable all session
    rag_down: bool = False                   # the local document store raises an error
    web_down: bool = False                   # Tavily web search raises an error

    def active(self) -> list[str]:
        """List enabled faults and pair each transient failure count with its error kind"""
        out = []
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name.endswith("_error"):
                continue  # described together with its *_fail_first count
            if f.name.endswith("_fail_first") and value:
                kind = getattr(self, f.name.replace("_fail_first", "_error"))
                out.append(f"{f.name}={value} ({kind})")
            elif value is True:
                out.append(f.name)
        return out

    @property
    def agent_internal(self) -> bool:
        """Identify agent-specific faults that have no equivalent in the single-agent baseline"""
        return bool(self.diagnostic_fail_first or self.recommendation_fail_first or self.recommendation_llm_down
                    or self.coordinator_llm_down or self.rag_down or self.web_down)


@dataclass(frozen=True)
class Expect:
    """Declare expected outputs and optional rules for one task
    The parser requires allowed statuses, while unset optional rules are skipped
    """

    status: tuple[str, ...] = ()                  # allowed final statuses
    findings_include: tuple[str, ...] = ()        # must be in the report's findings
    findings_exclude: tuple[str, ...] = ()        # must not be in the report's findings
    no_findings: bool | None = None               # True: the report must list no findings
    low_confidence_include: tuple[str, ...] = ()  # must be reported AND flagged low_confidence
    uncertainty_surfaced: bool | None = None      # True: some doubt must be visible (see eval_checks)
    urgency: tuple[str, ...] = ()                 # allowed urgency values ("none" = no urgency given)
    evidence: tuple[str, ...] = ()                # allowed evidence_status values ("none" = not set)
    rechecks: tuple[int, int] | None = None       # inclusive range of recheck rounds (multi-agent only)
    must_call: tuple[str, ...] = ()               # Coordinator tools that must have run (multi-agent only)
    must_not_call: tuple[str, ...] = ()           # Coordinator tools that must never run (multi-agent only)
    user_action_contains: tuple[str, ...] = ()    # words the re-upload instruction must contain
    no_advice: bool | None = None                 # True: no recommendation may be given (empty or the gap text)
    errors_expected: bool | None = None           # True/False: the report's error list must (not) be empty
    text_must_not_match: tuple[str, ...] = ()     # extra regexes the recommendation text must not match


@dataclass(frozen=True)
class Task:
    """Hold one case, its controlled study and plane scores, patient details, faults and expected behaviour"""
    id: str
    split: str             # "development" or "unseen"
    difficulty: str        # one of DIFFICULTIES
    category: str          # one of CATEGORIES
    title: str
    purpose: str           # why this case is in the test set
    scores: dict[str, float] = field(default_factory=dict)  # study-level scores; unlisted findings = BASE_SCORE
    plane_scores: dict[str, dict[str, float]] = field(default_factory=dict)  # per-plane overrides
    missing_planes: tuple[str, ...] = ()
    quality_flags: tuple[str, ...] = ()
    patient: PatientMetadata = field(default_factory=PatientMetadata)
    faults: Faults = field(default_factory=Faults)
    expect: Expect = field(default_factory=Expect)

    @property
    def study_id(self) -> str:
        # Safe for the backend's study_id rule ([A-Za-z0-9._-]) and obviously a test id
        """Create a stable test study identifier compatible with the backend identifier format"""
        return f"eval-{self.id.lower()}"

    def findings(self) -> dict[str, float]:
        """Expand sparse scores into a complete finding mapping using BASE_SCORE for omitted findings"""
        return {name: self.scores.get(name, BASE_SCORE) for name in FINDINGS}

    def prediction(self) -> PredictionResponse:
        """Build the normal prediction response shape from controlled task scores
        Apply per-plane overrides, omit missing planes and record their quality flags
        """
        study = self.findings()
        per_plane = [
            PlaneScores(plane=plane, scores=study | self.plane_scores.get(plane, {}))
            for plane in PLANES if plane not in self.missing_planes
        ]
        flags = list(self.quality_flags) + [f"missing_plane:{p}" for p in PLANES if p in self.missing_planes]
        return PredictionResponse(
            study_id=self.study_id, findings=study, per_plane=per_plane, laterality="left",
            laterality_source="dicom_tag", quality_flags=flags, model_version=FIXTURE_MODEL_VERSION,
            latency_ms=0.0,
        )


# --- Loading and validation ---

class TaskFileError(ValueError):
    """Report a task-file problem with enough location detail to find the invalid case"""


def _check_keys(data: dict, allowed: set[str], where: str) -> None:
    # Unknown keys are almost always typos ("finding_include"), which would silently disable a check
    """Reject unknown fields so misspelt expectation names cannot silently disable a check"""
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise TaskFileError(f"{where}: unknown field(s) {unknown}; allowed: {sorted(allowed)}")


def _check_names(names, where: str) -> None:
    """Require finding names to match the shared schema exactly"""
    unknown = sorted(set(names) - set(FINDINGS))
    if unknown:
        raise TaskFileError(f"{where}: unknown finding name(s) {unknown}; use exactly: {FINDINGS}")


def _check_values(values, allowed, where: str) -> None:
    """Reject values outside the allowed choices for this part of the task"""
    bad = sorted(set(values) - set(allowed))
    if bad:
        raise TaskFileError(f"{where}: {bad} not in {list(allowed)}")


def _parse_faults(data: dict, where: str) -> Faults:
    """Construct fault settings after checking field names and supported error kinds"""
    _check_keys(data, {f.name for f in fields(Faults)}, where)
    faults = Faults(**data)
    _check_values([faults.diagnostic_error, faults.recommendation_error], ERROR_KINDS, where)
    return faults


def _parse_expect(data: dict, where: str) -> Expect:
    """Convert JSON lists to tuples and validate expected statuses, findings, tools and ranges"""
    _check_keys(data, {f.name for f in fields(Expect)}, where)
    # JSON lists become tuples (the dataclass is frozen); true/false and numbers stay as they are
    values: dict = {key: tuple(value) if isinstance(value, list) else value for key, value in data.items()}
    expect = Expect(**values)
    _check_values(expect.status, STATUSES, f"{where}.status")
    _check_values(expect.urgency, URGENCIES, f"{where}.urgency")
    _check_values(expect.evidence, EVIDENCE, f"{where}.evidence")
    _check_values((*expect.must_call, *expect.must_not_call), TOOLS, f"{where} tools")
    _check_names((*expect.findings_include, *expect.findings_exclude, *expect.low_confidence_include), where)
    if expect.rechecks is not None and (len(expect.rechecks) != 2 or expect.rechecks[0] > expect.rechecks[1]):
        raise TaskFileError(f"{where}.rechecks must be [min, max]")
    if not expect.status:
        raise TaskFileError(f"{where}: every task must say which final status(es) are correct")
    return expect


def parse_task(data: dict, split: str) -> Task:
    """Validate task identity, split, categories and score bounds before building its structured form"""
    where = f"{TASK_FILES[split]} task {data.get('id', '?')}"
    allowed = {f.name for f in fields(Task)}
    _check_keys(data, allowed, where)
    for required in ("id", "split", "difficulty", "category", "title", "purpose"):
        if not data.get(required):
            raise TaskFileError(f"{where}: missing {required!r}")
    if data["split"] != split:
        raise TaskFileError(f"{where}: split is {data['split']!r} but the task is in the {split} file")
    _check_values([data["difficulty"]], DIFFICULTIES, f"{where}.difficulty")
    _check_values([data["category"]], CATEGORIES, f"{where}.category")
    scores = data.get("scores", {})
    _check_names(scores, f"{where}.scores")
    plane_scores = data.get("plane_scores", {})
    _check_values(plane_scores, PLANES, f"{where}.plane_scores")
    for plane, overrides in plane_scores.items():
        _check_names(overrides, f"{where}.plane_scores.{plane}")
    _check_values(data.get("missing_planes", []), PLANES, f"{where}.missing_planes")
    for value in [*scores.values(), *(v for o in plane_scores.values() for v in o.values())]:
        if not 0.0 <= value <= 1.0:
            raise TaskFileError(f"{where}: score {value} is not between 0 and 1")
    return Task(
        id=data["id"], split=split, difficulty=data["difficulty"], category=data["category"],
        title=data["title"], purpose=data["purpose"], scores=scores, plane_scores=plane_scores,
        missing_planes=tuple(data.get("missing_planes", [])), quality_flags=tuple(data.get("quality_flags", [])),
        patient=PatientMetadata(**data.get("patient", {})),
        faults=_parse_faults(data.get("faults", {}), f"{where}.faults"),
        expect=_parse_expect(data.get("expect", {}), f"{where}.expect"),
    )


def load_tasks(splits: tuple[str, ...] = ("development", "unseen"), directory: Path = TEST_CASES_DIR) -> list[Task]:
    """Read the requested split files, validate each case and reject duplicate task identifiers"""
    tasks: list[Task] = []
    for split in splits:
        raw = json.loads((directory / TASK_FILES[split]).read_text(encoding="utf-8"))
        tasks.extend(parse_task(item, split) for item in raw["tasks"])
    ids = [t.id for t in tasks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise TaskFileError(f"task ids used more than once: {duplicates}")
    return tasks


def build_replay_file(tasks: list[Task]) -> ReplayPredictionsFile:
    """Create controlled predictions for the real replay-mode Model Service
    Leave study_missing cases out so requesting those studies follows the HTTP 404 path
    """
    return ReplayPredictionsFile(
        model_version=FIXTURE_MODEL_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source="evaluation fixture: controlled scores from evaluation/test_cases (not trained-model output)",
        predictions=[t.prediction() for t in tasks if not t.faults.study_missing],
    )

