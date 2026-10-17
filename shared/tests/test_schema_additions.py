import pytest
from pydantic import ValidationError

from shared.schema import (DISCLAIMER, DiagnosticAgentResponse, FINDINGS, FinalReport, RecommendationResponse,
                           SessionStatus)


def test_disclaimer_is_always_present():
    report = FinalReport(session_id="s", study_id="x", status=SessionStatus.COMPLETED, disclaimer="Custom note.")
    assert DISCLAIMER in report.disclaimer and "Custom note." in report.disclaimer
    assert DISCLAIMER in FinalReport(session_id="s", study_id="x", status=SessionStatus.FAILED).disclaimer


def test_suggested_recheck_must_use_real_finding_names():
    with pytest.raises(ValidationError):
        RecommendationResponse(recommendation_text="", urgency="routine", evidence_status="grounded_local",
                               suggested_recheck=["Bakers_Cyst"])
    ok = RecommendationResponse(recommendation_text="", urgency="routine", evidence_status="grounded_local",
                                suggested_recheck=["Baker's"], concerns=["odd pattern"])
    assert ok.suggested_recheck == ["Baker's"]


def test_new_fields_are_optional_so_existing_code_keeps_working():
    scores = {name: 0.1 for name in FINDINGS}
    response = DiagnosticAgentResponse(study_id="x", findings=scores, per_finding_confidence=scores,
                                       overall_confidence=0.9, decision="finalize", retry_count=0, laterality="left")
    assert response.impression == ""
    assert isinstance(DiagnosticAgentResponse.model_fields["cross_plane_variance"].description, str)