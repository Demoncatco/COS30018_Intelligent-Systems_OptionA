"""Shared constants and tiny helpers. No heavy imports, so every other module can use this."""

# The 12 label columns of train.csv, in a fixed order (this is the model's output order).
ABNORMALITIES = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

# Imaging planes, in the order they are stacked into the model input.
PLANES = ["sagittal", "coronal", "axial"]

# Words in a Severity string (from extract_knee_findings.py) that mean "not abnormal".
NORMAL_WORDS = (
    "normal", "intact", "healthy", "unremarkable", "no abnormal", "absent",
    "none", "negative", "no evidence", "preserved", "not seen", "no ",
)


def is_normal(severity: str) -> bool:
    """True if a severity description says the structure is healthy."""
    s = str(severity).lower()
    return any(w in s for w in NORMAL_WORDS)
