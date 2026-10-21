# Example Coordinator requests for testing

from shared.schema import CoordinatorRequest, FINDINGS

# Strong finding with good local source coverage
STRONG_FINDING = CoordinatorRequest(
    finding={**{k: 0.05 for k in FINDINGS}, "ACL": 0.92},
    confidence=0.89,
    study_context={"laterality": "left"},
    task_type="initial",
)

# Finding with limited local source coverage
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