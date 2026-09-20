# Stub CoordinatorRequest fixtures, standing in for the real Coordinator Agent until it's ready. 

from shared.schema import CoordinatorRequest, FINDINGS

# Scenario A: strong finding, good local doc coverage expected (ACL has 6+ sources)
STRONG_FINDING = CoordinatorRequest(
    finding={**{k: 0.05 for k in FINDINGS}, "ACL": 0.92},
    confidence=0.89,
    study_context={"laterality": "left"},
    task_type="initial",
)

# Scenario B: thin local coverage expected (Synovitis has only 1 weak source)
WEAK_COVERAGE_FINDING = CoordinatorRequest(
    finding={**{k: 0.05 for k in FINDINGS}, "Synovitis": 0.71},
    confidence=0.65,
    study_context={"laterality": "right"},
    task_type="initial",
)

# Scenario C: everything near 0.5 - genuinely ambiguous, low overall confidence
AMBIGUOUS_FINDING = CoordinatorRequest(
    finding={k: 0.5 for k in FINDINGS},
    confidence=0.40,
    study_context={"laterality": "left"},
    task_type="initial",
)

# Scenario D: a Coordinator recheck request (Design B retry loop)
RECHECK_REQUEST = CoordinatorRequest(
    finding={**{k: 0.05 for k in FINDINGS}, "Baker's": 0.60},
    confidence=0.55,
    study_context={"laterality": "right"},
    task_type="recheck: find more evidence on Baker's cyst",
)