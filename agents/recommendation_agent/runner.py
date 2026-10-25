# Adapter between the Coordinator and the LangGraph recommendation loop.
#
# The Coordinator calls every agent through one function with a fixed contract (agents/coordinator/runners.py):
#     recommendation(request: CoordinatorRequest, *, session_id: str) -> RecommendationResponse
# The graph (graph.py) works on ONE finding name and returns a text recommendation plus the chunks it used.
# This file sits in between:
#   1. picks the findings worth looking up (probability >= REPORT_THRESHOLD, strongest first),
#   2. runs the graph once per finding, with the shared LLM client bound to this session, passing on the
#      Diagnostic Agent's impression and the patient details so the advice fits this patient,
#   3. turns the graph's results into a RecommendationResponse: text, citations, evidence_status,
#      urgency, concerns and suggested_recheck.
# Steps 1 and 3 are plain code, so the parts the Coordinator's rules depend on never come from the LLM.
#
# Plug it in:
#     from agents.recommendation_agent.runner import RecommendationAgent
#     recommendation = RecommendationAgent(llm=LLMClient())

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from shared.config import Settings, get_settings
from shared.llm import LLMClient
from shared.schema import (
    FINDINGS,
    CoordinatorRequest,
    EvidenceStatus,
    PatientMetadata,
    RecommendationResponse,
    UrgencyLevel,
)

from .graph import build_graph, initial_state
from .llm_client import make_qwen_caller
from .nodes.decide_action import DEFAULT_MAX_STEPS
from .nodes.synthesize import enforce_disclaimer

# At most this many findings are looked up per call. Each one costs several LLM calls, and the Coordinator
# stops waiting after TOOL_TIMEOUT_S, so looking up all 12 would time out.
DEFAULT_MAX_FINDINGS = 3

# --- Urgency rules (agree these with the team; they are triage defaults, not clinical guidance) ---
URGENT_FINDINGS = {"Fracture"}
SEE_DOCTOR_SOON_FINDINGS = {"ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Contusion", "Effusion", "Synovitis"}
# Medial OA, Lateral OA, PF OA and Baker's are usually long-term conditions: routine follow-up.

# --- Plausibility rules for concerns / suggested_recheck ---
# A reported finding this close above the threshold counts as borderline.
BORDERLINE_MARGIN = 0.1
# Findings that usually come with a fracture on MRI. A fracture with none of them is an unusual pattern.
FRACTURE_COMPANIONS = ("Contusion", "Effusion")

GENERAL_QUERY = "knee pain and swelling: when to see a doctor"
NO_FINDINGS_TEXT = (
    "No finding reached the reporting threshold on this scan. If knee pain, swelling, locking or giving way "
    "continues, this may indicate something the scan did not show, so please consult a clinician."
)

# Sessions whose looked-up findings are kept, so the senior's second review after a recheck does not repeat
# the same searches when nothing it depends on has changed. Oldest sessions are dropped first.
MAX_CACHED_SESSIONS = 32


@dataclass
class FindingResult:
    """What the graph produced for one finding."""

    finding: str
    probability: float
    recommendation: str
    local_chunks: list[dict] = field(default_factory=list)
    web_chunks: list[dict] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_evidence(self) -> bool:
        return bool(self.local_chunks or self.web_chunks)


class RecommendationAgent:
    """Callable that matches the Coordinator's RecommendationRunner contract. Raises on LLM failure."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        *,
        retrieve_fn: Callable | None = None,
        search_fn: Callable | None = None,
        llm_fn: Callable[[str, str | None], str] | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_findings: int = DEFAULT_MAX_FINDINGS,
    ) -> None:
        # llm_fn replaces the model entirely (tests and --demo); otherwise calls go through `llm`.
        self.llm = llm
        self.settings = settings or (llm.settings if llm is not None else get_settings())
        self.retrieve_fn = retrieve_fn
        self.search_fn = search_fn
        self.llm_fn = llm_fn
        self.max_steps = max_steps
        self.max_findings = max_findings
        self._cache: OrderedDict[str, dict[str, FindingResult]] = OrderedDict()

    # --- Entry point ---

    def __call__(self, request: CoordinatorRequest, *, session_id: str) -> RecommendationResponse:
        threshold = self.settings.report_threshold
        reported = sorted((n for n in FINDINGS if request.finding[n] >= threshold),
                          key=lambda n: -request.finding[n])
        looked_up = reported[: self.max_findings]

        trace = [f"task_type={request.task_type}; reported findings={reported or 'none'}"]
        if len(reported) > len(looked_up):
            trace.append(f"looked up the {len(looked_up)} strongest findings; skipped {reported[len(looked_up):]}")

        if not looked_up:
            results: list[FindingResult] = []
            text, citations, evidence = self._no_findings(trace)
        else:
            results = [self._research(name, request, session_id) for name in looked_up]
            for result in results:
                trace.extend(f"{result.finding}: {line}" for line in result.trace)
                trace.extend(f"{result.finding}: ERROR {error}" for error in result.errors)
            text = self._combine_text(results)
            citations = _unique_urls(c for r in results for c in (*r.local_chunks, *r.web_chunks))
            evidence = _evidence_status(results)

        concerns, recheck = plausibility_checks(request, self.settings)
        urgency = urgency_for(reported)
        trace.append(f"evidence={evidence.value}; urgency={urgency.value}; recheck requests={recheck or 'none'}")

        return RecommendationResponse(
            recommendation_text=text, citations=citations, urgency=urgency, evidence_status=evidence,
            trace=trace, concerns=concerns, suggested_recheck=recheck,
        )

    # --- Steps ---

    def _research(self, finding: str, request: CoordinatorRequest, session_id: str) -> FindingResult:
        # A second review reuses the earlier result only if the impression is unchanged. A recheck that changes
        # the impression gets a fresh recommendation, so the text never describes the old result.
        session = self._cache.setdefault(session_id, {})
        self._cache.move_to_end(session_id)
        while len(self._cache) > MAX_CACHED_SESSIONS:
            self._cache.popitem(last=False)
        key = f"{finding}|{request.impression}"
        if key in session:
            cached = session[key]
            return FindingResult(finding, request.finding[finding], cached.recommendation, cached.local_chunks,
                                 cached.web_chunks, ["reused the sources found earlier in this session"], [])

        llm_fn = self.llm_fn or make_qwen_caller(self.llm, session_id=session_id)
        graph = build_graph(llm=llm_fn, retrieve_fn=self.retrieve_fn, search_fn=self.search_fn)
        final = graph.invoke(
            initial_state(finding, _search_query(finding, request), self.max_steps,
                          impression=request.impression,
                          patient_context=patient_context(request.patient_metadata)),
            config={"recursion_limit": 2 * (self.max_steps + 2) + 10},
        )
        result = FindingResult(
            finding=finding, probability=request.finding[finding],
            recommendation=final.get("recommendation", ""),
            local_chunks=list(final.get("retrieved_chunks") or []),
            web_chunks=list(final.get("web_results") or []),
            trace=[line.strip() for line in final.get("trace", [])],
            errors=list(final.get("errors", [])),
        )
        session[key] = result
        return result

    def _no_findings(self, trace: list[str]) -> tuple[str, list[str], EvidenceStatus]:
        # Nothing to look up per finding, so the general "when to see a doctor" guidance is grounded instead.
        retrieve = self.retrieve_fn
        if retrieve is None:
            from .rag.retrieval import retrieve
        try:
            chunks = retrieve(GENERAL_QUERY, k=3, finding=None)
        except Exception as error:  # noqa: BLE001 - a missing index should not fail a normal study
            trace.append(f"general guidance lookup failed: {type(error).__name__}: {error}")
            chunks = []
        trace.append(f"no reported findings; general guidance from {len(chunks)} local chunk(s)")
        if not chunks:
            return NO_FINDINGS_TEXT, [], EvidenceStatus.NO_CREDIBLE_SOURCE_FOUND
        return NO_FINDINGS_TEXT, _unique_urls(chunks), EvidenceStatus.GROUNDED_LOCAL

    @staticmethod
    def _combine_text(results: list[FindingResult]) -> str:
        sections = [f"{r.finding} (model probability {r.probability:.2f})\n{r.recommendation.strip()}"
                    for r in results]
        text, _ = enforce_disclaimer("\n\n".join(sections))  # each section already has one; this is a backstop
        return text


# --- Rules (plain functions so they can be tested and explained on their own) ---

def urgency_for(reported: list[str]) -> UrgencyLevel:
    """Most urgent level among the reported findings."""
    found = set(reported)
    if found & URGENT_FINDINGS:
        return UrgencyLevel.URGENT
    if found & SEE_DOCTOR_SOON_FINDINGS:
        return UrgencyLevel.SEE_DOCTOR_SOON
    return UrgencyLevel.ROUTINE


def plausibility_checks(request: CoordinatorRequest, settings: Settings) -> tuple[list[str], list[str]]:
    """The senior's doubts about the junior's findings: (concerns, suggested_recheck)."""
    threshold = settings.report_threshold
    scores = request.finding
    reported = [n for n in FINDINGS if scores[n] >= threshold]
    concerns: list[str] = []
    recheck: list[str] = []

    # 1. A fracture usually shows bone bruising or joint fluid as well.
    if "Fracture" in reported and not any(scores[c] >= threshold for c in FRACTURE_COMPANIONS):
        concerns.append(f"Fracture {scores['Fracture']:.2f} without Contusion or Effusion is an unusual pattern.")
        recheck.append("Fracture")

    # 2. Low overall confidence plus findings that only just passed the threshold.
    if request.confidence < settings.low_confidence_threshold:
        borderline = [n for n in reported if scores[n] < threshold + BORDERLINE_MARGIN]
        if borderline:
            concerns.append(
                f"Diagnostic confidence {request.confidence:.2f} is below {settings.low_confidence_threshold:.2f} "
                f"and {', '.join(borderline)} only just passed the reporting threshold.")
            recheck.extend(n for n in borderline if n not in recheck)

    return concerns, [n for n in FINDINGS if n in recheck]


def patient_context(patient: PatientMetadata) -> str:
    """The patient details as one short line for the synthesis prompt, leaving out anything not given."""
    parts = []
    if patient.age is not None:
        parts.append(f"age {patient.age}")
    if patient.sex != "unknown":
        parts.append(patient.sex)
    if patient.symptoms.strip():
        parts.append(f"symptoms: {patient.symptoms.strip()}")
    if patient.injury_mechanism.strip():
        parts.append(f"injury: {patient.injury_mechanism.strip()}")
    return "; ".join(parts)


def _search_query(finding: str, request: CoordinatorRequest) -> str:
    # The patient's symptoms sharpen the search; they are only used as a search query, never as instructions.
    symptoms = request.patient_metadata.symptoms.strip()
    query = f"{finding} treatment and recovery"
    return f"{query}, patient reports: {symptoms[:200]}" if symptoms else query


def _evidence_status(results: list[FindingResult]) -> EvidenceStatus:
    if not any(r.has_evidence for r in results):
        return EvidenceStatus.NO_CREDIBLE_SOURCE_FOUND
    if any(r.web_chunks for r in results):
        return EvidenceStatus.GROUNDED_WEB
    return EvidenceStatus.GROUNDED_LOCAL


def _unique_urls(chunks) -> list[str]:
    return list(dict.fromkeys(c["source_url"] for c in chunks if c.get("source_url")))
