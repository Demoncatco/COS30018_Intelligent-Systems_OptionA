# The contract that will be used between the detection model and the other agents such as the diagnostic and recommendation agennts

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
        }
    )
