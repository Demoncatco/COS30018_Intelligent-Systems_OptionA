# Test whether the shared schema actually validates

import pytest
from pydantic import ValidationError

from shared.schema import (
    CoordinatorRequest,
    RecommendationResponse,
    UrgencyLevel,
    EvidenceStatus,
    FINDINGS,
)

FULL_FINDING = {k: 0.05 for k in FINDINGS}


def test_valid_request_parses():
    req = CoordinatorRequest(
        finding={**FULL_FINDING, "ACL": 0.8},
        confidence=0.9,
        study_context={"laterality": "left"},
        task_type="initial",
    )
    assert req.confidence == 0.9


def test_confidence_above_one_is_rejected():
    with pytest.raises(ValidationError):
        CoordinatorRequest(
            finding=FULL_FINDING,
            confidence=1.5,  # invalid - confidence must be 0.0-1.0
            study_context={},
            task_type="initial",
        )


def test_confidence_below_zero_is_rejected():
    with pytest.raises(ValidationError):
        CoordinatorRequest(
            finding=FULL_FINDING,
            confidence=-0.1,  # invalid
            study_context={},
            task_type="initial",
        )


def test_finding_missing_a_key_is_rejected():
    # Proves the reused _check_scores validator actually fires on CoordinatorRequest too
    incomplete = {k: 0.05 for k in FINDINGS if k != "Baker's"}  # drop one of the 12 on purpose
    with pytest.raises(ValidationError):
        CoordinatorRequest(
            finding=incomplete,
            confidence=0.9,
            study_context={},
            task_type="initial",
        )


def test_finding_with_wrong_key_name_is_rejected():
    # Guards against a mismatch an underscore-style name instead of the real column name.
    wrong_names = {k: 0.05 for k in FINDINGS if k != "Baker's"}
    wrong_names["Bakers_Cyst"] = 0.05  # wrong - real name is "Baker's"
    with pytest.raises(ValidationError):
        CoordinatorRequest(
            finding=wrong_names,
            confidence=0.9,
            study_context={},
            task_type="initial",
        )


def test_invalid_urgency_value_is_rejected():
    with pytest.raises(ValidationError):
        RecommendationResponse(
            recommendation_text="...",
            citations=[],
            urgency="super_urgent",  # not a real UrgencyLevel - typo simulation # pyright: ignore[reportArgumentType]
            evidence_status=EvidenceStatus.GROUNDED_LOCAL,
        )


def test_valid_response_parses():
    resp = RecommendationResponse(
        recommendation_text="May indicate an ACL tear. Consult a clinician.",
        citations=["https://orthoinfo.aaos.org/en/diseases--conditions/common-knee-injuries/"],
        urgency=UrgencyLevel.SEE_DOCTOR_SOON,
        evidence_status=EvidenceStatus.GROUNDED_LOCAL,
    )
    assert resp.urgency == "see_doctor_soon"


def test_missing_required_field_is_rejected():
    with pytest.raises(ValidationError):
        CoordinatorRequest(
            finding=FULL_FINDING,
            confidence=0.9,
            # missing task_type entirely
        ) # pyright: ignore[reportCallIssue]