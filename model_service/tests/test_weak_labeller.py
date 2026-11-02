# Weak labeller rules and output format. Uses short made-up reports, no dataset needed.

import pandas as pd
import pytest

from model_service.knee_model.labels import validate_weak_file
from model_service.knee_model.weak_labeller import agreement_with_gold, label_report, label_reports
from shared.schema import FINDINGS


@pytest.mark.parametrize("text, finding, expected", [
    ("Complete tear of the anterior cruciate ligament.", "ACL", 1),
    ("The ACL is intact.", "ACL", 0),
    ("Medial meniscus is not torn.", "Medial Meniscus", 0),
    ("Rotura de menisco interno.", "Medial Meniscus", 1),
    ("Menisco lateral de morfología y señal conservada, sin signos de rotura.", "Lateral Meniscus", 0),
    ("Ligamentos cruzados y colaterales dentro de límites normales.", "MCL", 0),
    ("Suspected partial tear of the ACL.", "ACL", -1),
    ("Low-grade sprain of the medial collateral ligament.", "MCL", -1),
    ("Moderate joint effusion.", "Effusion", 1),
    ("No joint effusion.", "Effusion", 0),
    ("Trace joint fluid effusion.", "Effusion", -1),
    ("Baker cyst: None.", "Baker's", 0),
    ("Full thickness cartilage loss along the medial femoral condyle.", "Medial OA", 1),
    ("Artrosis femorotibial medial.", "Medial OA", 1),
    ("Chondromalacia of the patella.", "PF OA", 1),
    ("Subchondral bone marrow edema in the lateral tibial plateau.", "Contusion", None),
    ("Bone contusion at proximal tibia.", "Contusion", 1),
    ("No fracture or bone contusion.", "Fracture", 0),
])
def test_rules(text, finding, expected):
    assert label_report(text)[finding] == expected


def test_unmentioned_findings_stay_blank():
    labels = label_report("Complete tear of the ACL.")
    assert labels["ACL"] == 1 and all(labels[f] is None for f in FINDINGS if f != "ACL")


def test_positive_clause_wins_over_negative():
    assert label_report("No effusion in the suprapatellar recess. Small effusion posteriorly.")["Effusion"] == 1


def test_output_passes_the_training_format_check():
    reports = pd.DataFrame({"StudyInstanceUID": ["a", "b"],
                            "Report": ["ACL tear. Moderate effusion.", "Unremarkable study, no effusion."]})
    frame = label_reports(reports)
    assert list(frame.columns) == ["study_id", *FINDINGS]
    cleaned = validate_weak_file(frame)
    assert cleaned.loc[0, "ACL"] == 1.0 and cleaned.loc[1, "Effusion"] == 0.0 and pd.isna(cleaned.loc[1, "ACL"])


def test_gold_agreement_table():
    row = {f: 0.0 for f in FINDINGS}
    train = pd.DataFrame([{"StudyInstanceUID": "g1", "Report": "Complete ACL tear. No effusion.", **row, "ACL": 1.0},
                          {"StudyInstanceUID": "g2", "Report": "ACL intact. Large effusion.", **row,
                           "Effusion": 0.0}])
    manifest = pd.DataFrame({"study_id": ["g1", "g2"], "label_source": ["gold", "gold"], "fallback_level": [1, 1]})
    table = agreement_with_gold(train, manifest).set_index("finding")
    assert table.loc["ACL", "accuracy"] == 1.0 and table.loc["ACL", "labelled"] == 2
    assert table.loc["Effusion", "accuracy"] == 0.5 and table.loc["Effusion", "precision"] == 0.0