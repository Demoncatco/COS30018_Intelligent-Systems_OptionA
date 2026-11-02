"""Keyword/rule weak labeller: radiology report text -> weak-label CSV in the agreed format (README 3.5).

Usage: python -m model_service.knee_model.weak_labeller --train-csv <train.csv> --out weak_labels.csv [--eval-gold]
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

from shared.schema import FINDINGS

from .labels import DEFAULT_MANIFEST, WEAK_ABSENT, WEAK_PRESENT, WEAK_UNCERTAIN, validate_weak_file

# --- Vocabulary (English, Spanish, Portuguese, Dutch, German; reports in the dataset mix these) ---

NEGATION = (r"\b(no|not|non|none|without|intact|normal|normally|unremarkable|preserved|negative|absent|"
            r"sin|ni|íntegro|íntegra|conservad[oa]s?|normales|"
            r"não|sem|preservad[oa]s?|"
            r"geen|zonder|intact|normaal|"
            r"kein|keine|keinen|ohne|unauffällig|regelrecht)\b")
HEDGE = (r"\b(possible|possibly|probable|probably|suspect\w*|suspicious|may|might|cannot be excluded|can't|"
         r"r/o|rule out|questionable|equivocal|"
         r"posible|probable|sospecha|sospechos[oa]|dudos[oa]|"
         r"possível|suspeita|"
         r"mogelijk|verdacht|waarschijnlijk|"
         r"fraglich|verdacht|möglich\w*|v\.a\.)\b")
INJURY = (r"(tear|torn|tearing|rupture|ruptured|disruption|sprain|injur\w*|avuls\w*|deficien\w*|"
          r"amputation|grade (2|3|ii|iii)|"
          r"rotura|roto|rota|ruptura|desgarro|lesión|lesion|esguince|distensión|amputación|"
          r"scheur|ruptuur|gescheurd|letsel|verstuiking|"
          r"riss|ruptur|läsion|zerrung)")
DEGENERATION = (r"(osteoarthrit\w*|arthros\w*|artros\w*|artrose|arthrose|chondr\w*|condr\w*|cartilag\w+ (loss|defect|"
                r"thinning|fissur\w*)|full[- ]thickness|osteophyt\w*|osteofit\w*|osteofyt\w*|kraakbeen\w*|knorpel\w*|"
                r"denudat\w*|úlcera condral|cartílago)")

TERMS = {
    "ACL": r"(\bACL\b|anterior cruciate|cruciate ligaments|\bLCA\b|cruzado anterior|ligamentos cruzados|"
           r"voorste kruisband|kruisbanden|vordere\w* kreuzband|kreuzbänder|\bVKB\b)",
    "MCL": r"(\bMCL\b|medial collateral|tibial collateral|collateral ligaments|colateral (medial|interno|tibial)|"
           r"colaterales|\bLCM\b|\bLCI\b|mediale collaterale|collaterale banden|innenband|seitenbänder)",
    "Medial Meniscus": r"(medial meniscus|meniscus medialis|menisco (medial|interno)|mediale meniscus|innenmeniskus|"
                       r"\bmenisci\b|meniscos|menisken)",
    "Lateral Meniscus": r"(lateral meniscus|meniscus lateralis|menisco (lateral|externo)|laterale meniscus|"
                        r"außenmeniskus|aussenmeniskus|\bmenisci\b|meniscos|menisken)",
    "Effusion": r"(effusion|derrame|hydrops|erguss|hemarthros\w*|lipohemarthros\w*|hemartros\w*|gewrichtsvocht)",
    "Synovitis": r"(synovitis|sinovitis|sinovite|synovial (thickening|hypertroph\w*|proliferation)|"
                 r"synoviale hypertrofie|engrosamiento sinovial)",
    "Baker's": r"(baker|popliteal cyst|quiste poplíteo|quistes poplíteos|cisto poplíteo|poplitea\w* cyste|bakercyste)",
    "Contusion": r"(contusion|bone bruise|marrow (o)?edema|bone (o)?edema|contusión|contusão|edema óseo|"
                 r"edema de (la )?médula|botoedeem|beenmergoedeem|knochenmark\w*ödem|kneuzing|bone marrow (o)?edema)",
    "Fracture": r"(fracture|fractura|fratura|fractuur|fraktur|avulsion fracture)",
}
# The term itself already means "present" (no injury word needed).
SELF_POSITIVE = {"Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"}

REGIONS = {
    "Medial OA": r"(medial (femoral condyle|tibial plateau|compartment|femorotibial|tibiofemoral|femoro-tibial)|"
                 r"(femorotibial|tibiofemoral|femoro-tibial) medial|compartimento (medial|interno)|"
                 r"cóndilo femoral (medial|interno)|platillo tibial (medial|interno)|mediaal femorotibiaal|"
                 r"mediale femorotibiale|medialen kompartiment)",
    "Lateral OA": r"(lateral (femoral condyle|tibial plateau|compartment|femorotibial|tibiofemoral|femoro-tibial)|"
                  r"(femorotibial|tibiofemoral|femoro-tibial) lateral|compartimento (lateral|externo)|"
                  r"cóndilo femoral (lateral|externo)|platillo tibial (lateral|externo)|lateraal femorotibiaal|"
                  r"laterale femorotibiale|lateralen kompartiment)",
    "PF OA": r"(patell\w*|trochle\w*|tróclea|rotulian\w*|rótula|femoropatel\w*|femoro-patel\w*|retropatellar)",
}
LOW_GRADE_EFFUSION = r"\b(trace|physiologic\w*|minimal|minima|mínimo|fisiológico)\b"
SMALL = r"\b(trace|tiny|small|minimal|pequeñ[oa]|discret[oa]|klein\w*)\b"
# Low-grade collateral sprains are often not counted as an MCL abnormality, so they stay unknown.
LOW_GRADE_SPRAIN = r"(low[- ]grade|grade (1|i|2|ii)\b|grado (1|i|2|ii)\b|partial|sprain|esguince|distensión|zerrung)"
# Degenerative (subchondral) marrow oedema is not a traumatic contusion.
DEGENERATIVE_OEDEMA = r"(subchondral|subcondral|subchondraal)"
CONTUSION_WORD = r"(contusi\w+|bruise|kneuzing)"
# Acute osteochondral injury is trauma, not osteoarthritis.
ACUTE_OSTEOCHONDRAL = r"(osteochondral (injury|impaction|fracture)|impaction)"

_SPLIT = re.compile(r"(?<!\d)[.;](?!\d)|\n|\s>\s|\s--\s")
_FLAGS = re.IGNORECASE | re.UNICODE


def clauses(report: str) -> list[str]:
    return [c.strip() for c in _SPLIT.split(str(report)) if c and c.strip()]


def _clause_label(finding: str, clause: str) -> int | None:
    """1 / 0 / -1 for one clause, or None if the clause says nothing about the finding."""
    if finding in REGIONS:
        if not (re.search(REGIONS[finding], clause, _FLAGS) and re.search(DEGENERATION, clause, _FLAGS)):
            return None
    elif not re.search(TERMS[finding], clause, _FLAGS):
        return None
    if re.search(NEGATION, clause, _FLAGS):
        return WEAK_ABSENT
    if re.search(HEDGE, clause, _FLAGS):
        return WEAK_UNCERTAIN
    if finding == "Effusion" and re.search(LOW_GRADE_EFFUSION, clause, _FLAGS):
        return WEAK_UNCERTAIN  # trace fluid: not clearly an effusion either way
    if finding == "Baker's" and re.search(SMALL, clause, _FLAGS):
        return WEAK_UNCERTAIN
    if finding == "Contusion" and re.search(DEGENERATIVE_OEDEMA, clause, _FLAGS) \
            and not re.search(CONTUSION_WORD, clause, _FLAGS):
        return None
    if finding in REGIONS and re.search(ACUTE_OSTEOCHONDRAL, clause, _FLAGS):
        return None
    if finding == "MCL" and re.search(INJURY, clause, _FLAGS) and re.search(LOW_GRADE_SPRAIN, clause, _FLAGS) \
            and not re.search(r"(complete|full[- ]thickness|grade (3|iii)\b|high[- ]grade)", clause, _FLAGS):
        return WEAK_UNCERTAIN
    if finding in SELF_POSITIVE or finding in REGIONS:
        return WEAK_PRESENT
    return WEAK_PRESENT if re.search(INJURY, clause, _FLAGS) else None


def label_report(report: str) -> dict[str, float | None]:
    """Any positive clause wins, then hedged, then negative; blank (None) if never mentioned."""
    out: dict[str, float | None] = {}
    parts = clauses(report)
    for finding in FINDINGS:
        votes = {v for v in (_clause_label(finding, c) for c in parts) if v is not None}
        for value in (WEAK_PRESENT, WEAK_UNCERTAIN, WEAK_ABSENT):
            if value in votes:
                out[finding] = value
                break
        else:
            out[finding] = None
    return out


def label_reports(reports: pd.DataFrame, id_col: str = "StudyInstanceUID", text_col: str = "Report") -> pd.DataFrame:
    rows = [{"study_id": sid, **label_report(text)} for sid, text in zip(reports[id_col], reports[text_col])]
    frame = pd.DataFrame(rows, columns=["study_id", *FINDINGS])
    frame[FINDINGS] = frame[FINDINGS].astype("Int64")  # whole numbers, blanks stay blank in the CSV
    return frame


def agreement_with_gold(train: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Runs the labeller on the 58 gold reports and compares with the expert labels, per finding."""
    gold_ids = set(manifest.loc[manifest["label_source"] == "gold", "study_id"])
    gold = train[train["StudyInstanceUID"].isin(gold_ids)]
    weak = validate_weak_file(label_reports(gold))  # 1.0 / 0.0 / NaN
    truth = gold.set_index("StudyInstanceUID")[FINDINGS].astype(float).gt(0).astype(float)
    weak = weak.set_index("study_id").loc[truth.index]
    rows = []
    for name in FINDINGS:
        known = weak[name].notna()
        pred, true = weak.loc[known, name], truth.loc[known, name]
        tp, fp = int(((pred == 1) & (true == 1)).sum()), int(((pred == 1) & (true == 0)).sum())
        fn = int(((pred == 0) & (true == 1)).sum())
        rows.append({"finding": name, "labelled": int(known.sum()), "coverage": round(known.mean(), 3),
                     "accuracy": round(float((pred == true).mean()), 3) if known.any() else None,
                     "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                     "recall": round(tp / (tp + fn), 3) if tp + fn else None})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-csv", required=True, help="Kaggle train.csv (StudyInstanceUID, Report, 12 findings)")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", default="weak_labels.csv")
    parser.add_argument("--eval-gold", metavar="CSV", nargs="?", const="weak_labeller_gold_agreement.csv",
                        help="Also score the rules on the gold reports and save the table")
    args = parser.parse_args()

    train, manifest = pd.read_csv(args.train_csv), pd.read_csv(args.manifest)
    weak_ids = set(manifest.loc[manifest["label_source"] == "weak", "study_id"])
    reports = train[train["StudyInstanceUID"].isin(weak_ids)]
    if len(reports) != len(weak_ids):
        raise SystemExit(f"manifest lists {len(weak_ids)} weak studies but train.csv has {len(reports)} of them")

    labels = label_reports(reports)
    validate_weak_file(labels)  # same check training runs, so a bad file never leaves this script
    labels.to_csv(args.out, index=False)
    counts = {v: int((labels[FINDINGS] == v).sum().sum()) for v in (1, 0, -1)}
    print(f"wrote {args.out}: {len(labels)} studies; cells present={counts[1]} absent={counts[0]} "
          f"hedged={counts[-1]} blank={int(labels[FINDINGS].isna().sum().sum())}")

    if args.eval_gold:
        table = agreement_with_gold(train, manifest)
        table.to_csv(args.eval_gold, index=False)
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()