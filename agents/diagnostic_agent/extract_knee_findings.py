#!/usr/bin/env python3
"""
extract_knee_findings.py

Reads knee MRI studies from train.csv (StudyInstanceUID + free-text Report, in any
language), asks a local MedGemma 4B model (served by Ollama) to identify which knee
structures/abnormalities the report mentions and how severe / healthy each one is,
and writes the result to a CSV:

    StudyInstanceUID, Abnormality, Severity

Abnormality values are always one of the train.csv label names:
    ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA,
    Effusion, Synovitis, Baker's, Contusion, Fracture

Setup:
    pip install pandas requests tqdm
    ollama pull medgemma:4b        # or whatever tag you use; see --model
    ollama serve                   # usually already running

Usage:
    python extract_knee_findings.py
    python extract_knee_findings.py --input train.csv --output findings.csv --limit 20
    python extract_knee_findings.py --model medgemma:4b --include-normal

The script is resumable: studies already present in the output file are skipped,
and every study is flushed to disk as soon as it is processed.
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import pandas as pd
import requests

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **kwargs):
        return it

# --------------------------------------------------------------------------- #
# Label vocabulary (matches the train.csv columns)
# --------------------------------------------------------------------------- #
ABNORMALITIES = [
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

# Guidance given to the model so non-English terms map to the right label.
LABEL_GUIDE = """\
- ACL: anterior cruciate ligament (ligamento cruzado anterior, LCA, vordere Kreuzband, voorste kruisband, LCA, ligament croisé antérieur)
- MCL: medial collateral ligament (ligamento colateral medial/interno/tibial, LCM, Innenband, mediale collaterale band, ligament collatéral médial)
- Medial Meniscus: medial/internal meniscus, incl. posterior horn / body / anterior horn (menisco medial/interno, Innenmeniskus, mediale meniscus, ménisque interne/médial)
- Lateral Meniscus: lateral/external meniscus (menisco lateral/externo, Außenmeniskus, laterale meniscus, ménisque externe/latéral)
- Medial OA: osteoarthritis / chondropathy / cartilage loss / osteophytes in the MEDIAL femorotibial compartment (artrosis femorotibial medial, mediale gonartrose, mediale artrose)
- Lateral OA: osteoarthritis / chondropathy / cartilage loss in the LATERAL femorotibial compartment
- PF OA: patellofemoral osteoarthritis / chondromalacia patellae / patellofemoral cartilage loss (artrosis femoropatelar, Retropatellararthrose, patellofemorale artrose)
- Effusion: joint effusion / fluid (derrame articular, derrame, Gelenkerguss, vochtuitstorting, épanchement)
- Synovitis: synovitis / synovial thickening or hypertrophy / Hoffa's fat pad oedema (sinovitis, Synovitis, synovitis)
- Baker's: Baker's cyst / popliteal cyst (quiste de Baker, quiste poplíteo, Baker-Zyste, Bakerse cyste, kyste de Baker)
- Contusion: bone marrow oedema / bone bruise / contusion (contusión ósea, edema óseo, Knochenmarködem, beenmergoedeem, contusion osseuse)
- Fracture: fracture, stress fracture, subchondral fracture, avulsion (fractura, Fraktur, fractuur)\
"""

SYSTEM_PROMPT = f"""You are a careful musculoskeletal radiology assistant.
You are given a knee MRI radiology report. It may be written in ANY language
(Spanish, Dutch, German, French, English, ...), may contain a clinical question,
technique section, findings and impression, and may be truncated or contain
placeholders such as [DATE].

Task: list every knee structure or abnormality from the fixed list below that the
report actually comments on, and describe its severity or health state.

Allowed abnormality labels (use these EXACT strings, never anything else):
{LABEL_GUIDE}

Rules:
1. Only report what the report explicitly states about the scan. Do NOT guess.
2. A clinical question or referral reason ("Diagnostische vraagstelling: meniscusscheur?")
   is NOT a finding. Ignore it unless the findings/impression confirm it.
3. Include structures described as normal/intact, with severity "Normal"
   (e.g. "ligamentos cruzados normales" -> ACL: Normal). Do not include structures
   the report does not mention.
4. "severity" must be a SHORT English phrase (max ~8 words) that states the state and
   grade, e.g. "Complete tear", "Partial tear, grade 2", "Intact / normal",
   "Mild", "Moderate", "Severe, full-thickness cartilage loss", "Small effusion",
   "Absent". Use the report's own grading when given.
5. One entry per label. If the report mentions a label several times, merge them
   into one entry and keep the most significant description.
6. If the report is empty, unreadable, or mentions none of the labels, return an
   empty findings list.
7. Respond ONLY with JSON matching the required schema.\
"""

# JSON schema for Ollama structured outputs (constrains the model's output).
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "abnormality": {"type": "string", "enum": ABNORMALITIES},
                    "severity": {"type": "string"},
                },
                "required": ["abnormality", "severity"],
            },
        }
    },
    "required": ["findings"],
}

NORMAL_WORDS = (
    "normal", "intact", "healthy", "unremarkable", "no abnormal", "absent",
    "none", "negative", "no evidence", "preserved", "not seen", "no ",
)


# --------------------------------------------------------------------------- #
# Ollama helpers
# --------------------------------------------------------------------------- #
def check_ollama(host: str, model: str) -> None:
    """Fail early with a helpful message if Ollama or the model is unavailable."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Cannot reach Ollama at {host}: {e}\nStart it with `ollama serve`.")
    names = [m.get("name", "") for m in r.json().get("models", [])]
    if model not in names and f"{model}:latest" not in names:
        print(
            f"WARNING: model '{model}' not found in Ollama. Installed models: {names or 'none'}.\n"
            f"         Pull it with `ollama pull {model}` or pass --model <tag>.",
            file=sys.stderr,
        )


def query_model(report: str, host: str, model: str, num_ctx: int, timeout: int) -> list[dict]:
    payload = {
        "model": model,
        "stream": False,
        "format": RESPONSE_SCHEMA,
        "options": {"temperature": 0, "num_ctx": num_ctx},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"REPORT:\n\"\"\"\n{report}\n\"\"\"\n\nReturn the JSON."},
        ],
    }
    r = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    return parse_findings(content)


def parse_findings(content: str) -> list[dict]:
    """Parse model output into a clean, de-duplicated list of findings."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fall back to the first {...} block if the model wrapped the JSON in text.
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            raise
        data = json.loads(m.group(0))

    findings = data.get("findings", []) if isinstance(data, dict) else []
    canonical = {a.lower(): a for a in ABNORMALITIES}
    merged: dict[str, str] = {}
    for f in findings:
        label = canonical.get(str(f.get("abnormality", "")).strip().lower())
        if not label:
            continue  # ignore anything outside the label vocabulary
        sev = " ".join(str(f.get("severity", "")).split()) or "Unspecified"
        prev = merged.get(label)
        # Prefer an abnormal description over a "normal" one if the model repeated a label.
        if prev is None or (is_normal(prev) and not is_normal(sev)):
            merged[label] = sev
    return [{"abnormality": k, "severity": v} for k, v in merged.items()]


def is_normal(severity: str) -> bool:
    s = severity.lower()
    return any(w in s for w in NORMAL_WORDS)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def load_done_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    try:
        return set(pd.read_csv(path, usecols=["StudyInstanceUID"])["StudyInstanceUID"].astype(str))
    except Exception:
        return set()


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract knee findings from reports with MedGemma via Ollama.")
    ap.add_argument("--input", default="train.csv")
    ap.add_argument("--output", default="findings.csv")
    ap.add_argument("--model", default="medgemma:4b", help="Ollama model tag (check `ollama list`)")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N studies (for testing)")
    ap.add_argument("--max-chars", type=int, default=6000, help="Truncate very long reports to this many characters")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=180, help="Seconds per request")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--abnormal-only", action="store_true",
                    help="Drop findings whose severity indicates a normal/healthy structure "
                         "(default: normal findings are kept, labelled e.g. 'Intact / normal')")
    args = ap.parse_args()

    check_ollama(args.host, args.model)

    df = pd.read_csv(args.input, usecols=["StudyInstanceUID", "Report"])
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    if args.limit:
        df = df.head(args.limit)

    done = load_done_ids(args.output)
    todo = df[~df["StudyInstanceUID"].isin(done)]
    print(f"{len(df)} studies in input, {len(done)} already done, {len(todo)} to process.")

    new_file = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    errors_path = os.path.splitext(args.output)[0] + "_errors.csv"

    with open(args.output, "a", newline="", encoding="utf-8") as out, \
         open(errors_path, "a", newline="", encoding="utf-8") as err:
        writer = csv.writer(out)
        err_writer = csv.writer(err)
        if new_file:
            writer.writerow(["StudyInstanceUID", "Abnormality", "Severity"])
            out.flush()

        for row in tqdm(todo.itertuples(index=False), total=len(todo), desc="Studies"):
            uid = row.StudyInstanceUID
            report = "" if pd.isna(row.Report) else str(row.Report).strip()[: args.max_chars]

            findings, last_err = None, None
            if report:
                for attempt in range(args.retries + 1):
                    try:
                        findings = query_model(report, args.host, args.model, args.num_ctx, args.timeout)
                        break
                    except Exception as e:  # network, timeout, bad JSON ...
                        last_err = e
                        time.sleep(1 + attempt)
                if findings is None:
                    err_writer.writerow([uid, repr(last_err)])
                    err.flush()
                    continue  # not written to output, so a re-run will retry it
            else:
                findings = []

            if args.abnormal_only:
                findings = [f for f in findings if not is_normal(f["severity"])]

            if findings:
                for f in findings:
                    writer.writerow([uid, f["abnormality"], f["severity"]])
            else:
                # Written so the study counts as processed on a re-run.
                writer.writerow([uid, "None detected", "No relevant findings in report"])
            out.flush()

    print(f"Done. Results: {args.output}   (failures, if any: {errors_path})")


if __name__ == "__main__":
    main()
