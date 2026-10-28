# Offline RAG preparation: split source text into overlapping chunks
# Paragraphs and sentence boundaries keep related text together where possible
# Source URLs and finding tags stay attached so retrieved evidence can be filtered and cited

import re
from pathlib import Path

CHUNK_TARGET_CHARS = 800
CHUNK_OVERLAP_CHARS = 150
CHUNK_MIN_CHARS = 200

# Split longer paragraphs into sentences before packing chunks
LONG_PARAGRAPH_THRESHOLD = int(CHUNK_TARGET_CHARS * 1.5)

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def slugify(title: str) -> str:
    """Use the same filename convention as fetching so source text can be found reliably"""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug[:60]


def split_into_paragraphs(text: str) -> list[str]:
    """Split on blank lines and remove empty paragraphs"""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def split_into_sentences(paragraph: str) -> list[str]:
    """Use punctuation and a following capital or digit as a simple sentence-boundary heuristic"""
    sentences = SENTENCE_SPLIT_PATTERN.split(paragraph)
    return [s.strip() for s in sentences if s.strip()]


def units_for_document(text: str) -> list[str]:
    """Keep ordinary paragraphs together and divide long paragraphs into sentence units"""
    units = []
    for paragraph in split_into_paragraphs(text):
        if len(paragraph) > LONG_PARAGRAPH_THRESHOLD:
            units.extend(split_into_sentences(paragraph))
        else:
            units.append(paragraph)
    return units


def hard_split_long_unit(unit: str) -> list[str]:
    """Pack an oversized unit into word-based pieces
    A single word longer than the target remains intact rather than being cut mid-word
    """
    words = unit.split(" ")
    pieces, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > CHUNK_TARGET_CHARS and current:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def overlap_prefix(previous_chunk_text: str) -> str:
    """Reuse the end of the previous chunk, moving the start past the first space in the slice"""
    if len(previous_chunk_text) <= CHUNK_OVERLAP_CHARS:
        return previous_chunk_text
    tail = previous_chunk_text[-CHUNK_OVERLAP_CHARS:]
    space_idx = tail.find(" ")
    return tail[space_idx + 1:] if space_idx != -1 else tail


def pack_units_into_chunks(units: list[str]) -> list[str]:
    """Pack units toward the target size with overlap between neighbouring chunks
    Overlap and the final small-chunk merge can make chunks larger than the target
    """
    # Split oversized units before packing, preserving complete words
    expanded = []
    for unit in units:
        if len(unit) > CHUNK_TARGET_CHARS:
            expanded.extend(hard_split_long_unit(unit))
        else:
            expanded.append(unit)
    units = expanded

    chunks: list[str] = []
    current = ""

    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= CHUNK_TARGET_CHARS:
            current = candidate
            continue

        # Start the next chunk with some overlap from the current one
        if current:
            chunks.append(current)
        prefix = overlap_prefix(current) if current else ""
        current = f"{prefix} {unit}".strip() if prefix else unit

    if current:
        chunks.append(current)

    # Merge a too-short trailing chunk into the one before it
    if len(chunks) > 1 and len(chunks[-1]) < CHUNK_MIN_CHARS:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
        chunks.pop()

    return chunks


def chunk_document_text(text: str) -> list[str]:
    """Run paragraph and sentence preparation, then pack the resulting units into chunks"""
    units = units_for_document(text)
    if not units:
        return []
    return pack_units_into_chunks(units)


def chunk_entry(entry: dict, raw_dir: Path) -> list[dict] | None:
    """Read one cached source and attach finding tags, citation fields and chunk positions
    Return None for a missing file so it can be reported separately from an empty document
    """
    slug = slugify(entry["title"])
    raw_path = raw_dir / f"{slug}.txt"
    if not raw_path.exists():
        return None

    text = raw_path.read_text(encoding="utf-8")
    pieces = chunk_document_text(text)

    return [
        {
            "chunk_id": f"{slug}__{i:03d}",
            "source_title": entry["title"],
            "source_url": entry["url"],
            "findings": entry["findings"],
            "chunk_index": i,
            "chunk_count": len(pieces),
            "text": piece,
        }
        for i, piece in enumerate(pieces)
    ]


def chunk_manifest(manifest: list[dict], raw_dir: Path) -> tuple[list[dict], list[str]]:
    """Collect chunks for available manifest sources and return the titles of missing files"""
    all_chunks: list[dict] = []
    missing: list[str] = []

    for entry in manifest:
        chunks = chunk_entry(entry, raw_dir)
        if chunks is None:
            missing.append(entry["title"])
        else:
            all_chunks.extend(chunks)

    return all_chunks, missing


if __name__ == "__main__":
    # Check chunking with the files in data/raw/
    import json

    manifest_path = Path(__file__).parent / "manifest.json"
    raw_dir = Path(__file__).parent / "data" / "raw"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks, missing = chunk_manifest(manifest, raw_dir)

    print(f"{len(chunks)} chunks from {len(manifest) - len(missing)}/{len(manifest)} documents")
    if missing:
        print(f"\n{len(missing)} entries have no raw/ file yet (run fetch_documents.py first):")
        for title in missing:
            print(f"  - {title}")
