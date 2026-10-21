# Fetch the pages in manifest.json and save their text locally.

import json
import re
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import requests
import trafilatura
from pypdf import PdfReader

MANIFEST_PATH = Path(__file__).parent / "manifest.json"
RAW_DIR = Path(__file__).parent / "data" / "raw"
MANUAL_FETCH_LOG = Path(__file__).parent / "data" / "manual_fetch_needed.txt"
REQUEST_DELAY_SECONDS = 2
HEADERS = {"User-Agent": "Mozilla/5.0 (educational research project)"}

PMC_ID_PATTERN = re.compile(r"(PMC\d+)")
PDF_HREF_PATTERN = re.compile(r'href="([^"]*\.pdf)"', re.IGNORECASE)
PDF_MAGIC_BYTES = b"%PDF-"


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug[:60]


def is_pmc_or_ncbi_books(url: str) -> bool:
    return (
        "ncbi.nlm.nih.gov/pmc/articles/" in url
        or "pmc.ncbi.nlm.nih.gov/articles/" in url
        or "ncbi.nlm.nih.gov/books" in url
    )


def canonical_pmc_pdf_guess(url: str) -> str | None:
    if "ncbi.nlm.nih.gov/books" in url:
        return None
    match = PMC_ID_PATTERN.search(url)
    if not match:
        return None
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{match.group(1)}/pdf/"


def find_real_pdf_link(article_html: str, article_url: str) -> str | None:
    match = PDF_HREF_PATTERN.search(article_html)
    if not match:
        return None
    return urljoin(article_url, match.group(1))


def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def try_pdf_fallback(entry: dict, article_html: str) -> str | None:
    url = entry["url"]
    candidates = []

    real_link = find_real_pdf_link(article_html, url)
    if real_link:
        candidates.append(real_link)

    guessed = canonical_pmc_pdf_guess(url)
    if guessed and guessed not in candidates:
        candidates.append(guessed)

    for pdf_url in candidates:
        print(f"  [retry] {entry['title']} - trying PDF at {pdf_url} ...")
        try:
            pdf_response = requests.get(pdf_url, headers=HEADERS, timeout=20)
            pdf_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"  [retry-fail] {pdf_url} - {e}")
            continue

        if not pdf_response.content.startswith(PDF_MAGIC_BYTES):
            print(f"  [retry-fail] {pdf_url} - response wasn't a real PDF (likely a landing/redirect page)")
            continue

        text = extract_pdf_text(pdf_response.content)
        if text and len(text) >= 200:
            return text
        print(f"  [retry-fail] {pdf_url} - PDF parsed but extracted text still too short")

    return None


def fetch_one(entry: dict) -> str:
    slug = slugify(entry["title"])
    out_path = RAW_DIR / f"{slug}.txt"
    if out_path.exists():
        print(f"  [skip] {slug}.txt already exists")
        return "skip"

    try:
        response = requests.get(entry["url"], headers=HEADERS, timeout=15)
        response.raise_for_status()
        text = trafilatura.extract(response.text)

        # Try the PDF if the page has too little text.
        if (not text or len(text) < 200) and is_pmc_or_ncbi_books(entry["url"]):
            text = try_pdf_fallback(entry, response.text)

        if not text or len(text) < 200:
            print(f"  [FAIL] {entry['title']} - extracted text still too short, needs manual review")
            return "fail"

        out_path.write_text(text, encoding="utf-8")
        print(f"  [ok]   {entry['title']} -> {slug}.txt ({len(text)} chars)")
        return "ok"

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            print(f"  [FAIL] {entry['title']} - 403 Forbidden (site likely blocks automated requests)")
        else:
            print(f"  [FAIL] {entry['title']} - {e}")
        return "fail"
    except Exception as e:
        print(f"  [FAIL] {entry['title']} - {e}")
        return "fail"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    print(f"Fetching {len(manifest)} documents...\n")
    counts = {"ok": 0, "skip": 0, "fail": 0}
    failed_entries = []

    for entry in manifest:
        status = fetch_one(entry)
        counts[status] += 1
        if status == "fail":
            failed_entries.append(entry)
        time.sleep(REQUEST_DELAY_SECONDS)

    fetched_total = counts["ok"] + counts["skip"]
    print(f"\nDone: {fetched_total}/{len(manifest)} available ({counts['ok']} fetched now, {counts['skip']} already had files).")

    if failed_entries:
        MANUAL_FETCH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with MANUAL_FETCH_LOG.open("w", encoding="utf-8") as f:
            f.write("Pages that need manual copy-paste - automated fetch failed or was blocked:\n\n")
            for entry in failed_entries:
                f.write(f"- {entry['title']}\n  {entry['url']}\n\n")
        print(f"\n{len(failed_entries)} failed - logged to {MANUAL_FETCH_LOG}")


if __name__ == "__main__":
    main()
