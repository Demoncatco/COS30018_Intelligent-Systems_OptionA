# Build a local search index from the fetched document chunks.

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from chunker import chunk_manifest

MANIFEST_PATH = Path(__file__).parent / "manifest.json"
RAW_DIR = Path(__file__).parent / "data" / "raw"
INDEX_DIR = Path(__file__).parent / "data" / "index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
METADATA_PATH = INDEX_DIR / "metadata.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


def collect_chunks() -> tuple[list[dict], list[str]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return chunk_manifest(manifest, RAW_DIR)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    # Normalize the vectors so retrieval can use dot products for cosine similarity
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def save_index(embeddings: np.ndarray, chunk_metadata: list[dict]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    METADATA_PATH.write_text(json.dumps(chunk_metadata, indent=2), encoding="utf-8")


def main():
    print("Chunking documents...")
    chunks, missing = collect_chunks()

    if not chunks:
        print("No chunks to embed - run fetch_documents.py first.")
        return

    print(f"  {len(chunks)} chunks ready to embed")
    if missing:
        print(f"  ({len(missing)} manifest entries skipped - no raw/ file yet, run fetch_documents.py)")

    print(f"\nLoading embedding model ({EMBEDDING_MODEL_NAME}) - this downloads "
          f"the model the first time you run it...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"\nEmbedding {len(chunks)} chunks...")
    start = time.time()
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(model, texts)
    elapsed = time.time() - start

    save_index(embeddings, chunks)

    print(f"\nDone in {elapsed:.1f}s: {embeddings.shape[0]} vectors x {embeddings.shape[1]} dims")
    print(f"  -> {EMBEDDINGS_PATH}")
    print(f"  -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
