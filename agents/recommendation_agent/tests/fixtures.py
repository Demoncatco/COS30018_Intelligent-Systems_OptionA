# Example CoordinatorRequest values covering strong, weak, uncertain and recheck scenarios
# Each request supplies a complete finding-score mapping to satisfy the shared schema

from shared.schema import CoordinatorRequest, FINDINGS

# Strong ACL score used to exercise a confident positive request
STRONG_FINDING = CoordinatorRequest(
    finding={**{k: 0.05 for k in FINDINGS}, "ACL": 0.92},
    confidence=0.89,
    study_context={"laterality": "left"},
    task_type="initial",
)

# Synovitis example used to exercise limited-evidence scenarios
WEAK_COVERAGE_FINDING = CoordinatorRequest(
    finding={**{k: 0.05 for k in FINDINGS}, "Synovitis": 0.71},
    confidence=0.65,
    study_context={"laterality": "right"},
    task_type="initial",
)

# Uncertain findings with low overall confidence
AMBIGUOUS_FINDING = CoordinatorRequest(
    finding={k: 0.5 for k in FINDINGS},
    confidence=0.40,
    study_context={"laterality": "left"},
    task_type="initial",
)

# Request for more evidence
RECHECK_REQUEST = CoordinatorRequest(
    finding={**{k: 0.05 for k in FINDINGS}, "Baker's": 0.60},
    confidence=0.55,
    study_context={"laterality": "right"},
    task_type="recheck: find more evidence on Baker's cyst",
)