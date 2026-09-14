# This file runs the stub FastAPI application and checks that both endpoints (/health and /predict) return responses that match the contract established in shared/schema.py
# This file verifies shape and structure, value ranges, all 12 anomaly findings and the 3 planes (sagittal, coronal and axial), and that invalid requirests and responses are rejected.
# This file does not test prediction accuracy as the stub model generates random results.

# This file is a set of automated checks that prove that the stub server is behaving in the way that the other agents should expect it to.
# These tests can all be ran via pytest command from the repositories root.
# This ensures that we can test in the case that any changes to the schema or the API in the future that may break the contract can be identified as one of these tests will fail

# pytest finds any function in the file that starts with test_ and runs it.
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from model_service.api.main import MODEL_VERSION, app
from shared.schema import FINDINGS, PLANES, PredictionResponse

client = TestClient(app)

# Health Check. Sends a GET to /health, checks that status is 200 (this is HTTP's success), then response.json converts the reply JSON text to python dictionary
def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    # Below asserts check that the ok and version matches.
    assert body["status"] == "ok"
    assert body["model_version"] == MODEL_VERSION


# Valid Prediction check. Sends a POST to /predict with a body of {"study_id": "study-001"}
def test_predict_returns_valid_prediction_response():
    response = client.post("/predict", json={"study_id": "study-001"})
    assert response.status_code == 200
    # Re-parses the JSON through the shared model which proves that the wire format matches the contract the agents will import.
    parsed = PredictionResponse.model_validate(response.json())
    assert parsed.study_id == "study-001"
    assert parsed.model_version == MODEL_VERSION
    assert parsed.latency_ms >= 0

# Posts a request and checks that the keys of the findings are exactly equal to the 12 names in FINDINGS.
def test_predict_contains_all_twelve_findings():
    body = client.post("/predict", json={"study_id": "study-002"}).json()
    assert set(body["findings"]) == set(FINDINGS) # "set" disregards the order as it is not required.
    assert len(body["findings"]) == 12 # Checks the the count of findings is equal to 12

# Loops between every headline findings and every per-plane score to ensure that they are inbetween 0 and 1.
def test_predict_probabilities_are_between_zero_and_one():
    body = client.post("/predict", json={"study_id": "study-003"}).json()
    for name, prob in body["findings"].items():
        assert 0.0 <= prob <= 1.0, f"{name} out of range: {prob}"
    for plane in body["per_plane"]:
        for name, prob in plane["scores"].items():
            assert 0.0 <= prob <= 1.0, f"{plane['plane']}/{name} out of range: {prob}"

# Pulls out the plane name for every per-plane entry, then sorts them and checks that they match the sorted PLANES. Then confirms that every plane has all 12 findings
def test_predict_reports_all_three_planes():
    body = client.post("/predict", json={"study_id": "study-004"}).json()
    planes = [entry["plane"] for entry in body["per_plane"]]
    assert sorted(planes) == sorted(PLANES)
    # Loop ensures that every per_plane entry has all 12 findings.
    for entry in body["per_plane"]:
        assert set(entry["scores"]) == set(FINDINGS) # Order does not matter, therefore set is used.

# Checks that the stub model returns the same findings and per-plane entires for the same study_id
def test_predict_is_deterministic_per_study_id():
    # Same study_id has the same numbers, so agents can be built and tested with repeatable test cases.
    first = client.post("/predict", json={"study_id": "repeatable"}).json()
    second = client.post("/predict", json={"study_id": "repeatable"}).json()
    assert first["findings"] == second["findings"] # Checks findings of first run are same as findings of second run for same study_id
    assert first["per_plane"] == second["per_plane"] # Checks per_plane entires of first run are the same as per_plane entires of second run for same study_id

    other = client.post("/predict", json={"study_id": "different"}).json()
    assert other["findings"] != first["findings"]

# Sends an empty JSON body to ensure error is handled correctly.
def test_predict_rejects_missing_study_id():
    response = client.post("/predict", json={})
    assert response.status_code == 422

# Checks that pydantic rejects a good response that has been changed to out of range of 0 - 1 for any findings, and or if the response is missing anything
def test_schema_rejects_bad_probabilities():
    good = client.post("/predict", json={"study_id": "study-005"}).json()

    # Copies the dictionary and changes the value of ACL to 1.3. Copies all other finding scores for the findings.
    too_big = {**good, "findings": {**good["findings"], "ACL": 1.3}}
    with pytest.raises(ValidationError):
        PredictionResponse.model_validate(too_big) # Code inside the with statement must run otherwise the test fails.

    # Builds a copy of the dictionary where ACL is removed entirely. Keeps everything else except for ACL.
    missing = {
        **good,
        "findings": {k: v for k, v in good["findings"].items() if k != "ACL"},
    }
    with pytest.raises(ValidationError):
        PredictionResponse.model_validate(missing) # Asserts that this too, is rejected correctly.
