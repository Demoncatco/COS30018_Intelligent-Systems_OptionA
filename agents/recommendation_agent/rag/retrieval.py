# Search the local index created by build_index.py


import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).parent / "data" / "index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
METADATA_PATH = INDEX_DIR / "metadata.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None
_embeddings: np.ndarray | None = None
_metadata: list[dict] | None = None


class IndexNotBuiltError(RuntimeError):
    """The local index is missing; run build_index.py first."""


def _load_index() -> tuple[np.ndarray, list[dict]]:
    global _embeddings, _metadata
    if _embeddings is None or _metadata is None:
        if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
            raise IndexNotBuiltError(
                f"No index found at {INDEX_DIR} - run build_index.py first."
            )
        _embeddings = np.load(EMBEDDINGS_PATH)
        assert _embeddings is not None
        _metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        assert _metadata is not None
        if _embeddings.shape[0] != len(_metadata):
            raise RuntimeError(
                f"Index is corrupted: {_embeddings.shape[0]} embedding rows but "
                f"{len(_metadata)} metadata entries - rerun build_index.py."
            )
    return _embeddings, _metadata


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def retrieve(query: str, k: int = 3, finding: str | None = None) -> list[dict]:
    """Return the most similar chunks, highest score first.
    
    If a finding is supplied, search only chunks tagged with that finding."""
    embeddings, metadata = _load_index()
    model = _load_model()

    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    if finding is not None:
        candidate_idx = [i for i, m in enumerate(metadata) if finding in m["findings"]]
        if not candidate_idx:
            return []
        candidate_embeddings = embeddings[candidate_idx]
    else:
        candidate_idx = list(range(len(metadata)))
        candidate_embeddings = embeddings

    # Both vectors are normalized, so their dot product is cosine similarity
    scores = candidate_embeddings @ query_vec

    top_k = min(k, len(candidate_idx))
    top_local = np.argsort(scores)[::-1][:top_k]

    results = []
    for local_i in top_local:
        global_i = candidate_idx[local_i]
        entry = dict(metadata[global_i])
        entry["score"] = float(scores[local_i])
        results.append(entry)
    return results


def _print_results(query: str, results: list[dict]) -> None:
    print(f'\nTop {len(results)} results for: "{query}"\n')
    for rank, r in enumerate(results, start=1):
        print(f"{rank}. [{r['score']:.3f}] {r['source_title']}  ({', '.join(r['findings'])})")
        print(f"   {r['source_url']}")
        preview = r["text"][:200].replace("\n", " ")
        print(f"   \"{preview}...\"\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python retrieval.py "your query here" [finding]')
        print('Example: python retrieval.py "ACL tear treatment"')
        print('Example: python retrieval.py "joint swelling" Effusion')
        sys.exit(1)

    query_arg = sys.argv[1]
    finding_arg = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        results = retrieve(query_arg, k=3, finding=finding_arg)
    except IndexNotBuiltError as e:
        print(f"Error: {e}")
        sys.exit(1)

    _print_results(query_arg, results)
