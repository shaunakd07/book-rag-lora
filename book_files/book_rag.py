import os
import argparse
import pickle
from typing import List, Dict, Iterable, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def iter_text_files(input_dir: str) -> Iterable[str]:
    for name in sorted(os.listdir(input_dir)):
        if name.lower().endswith(".txt"):
            yield os.path.join(input_dir, name)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def build_rag_index(
    input_dir: str,
    out_dir: str,
    embed_model: str,
    chunk_size: int,
    overlap: int,
) -> Tuple[str, str, str]:
    """
    Builds:
      - all_chunks.pkl : List[str]
      - chunk_metadata.pkl : List[dict]
      - rag_index.faiss : FAISS index aligned with all_chunks ordering

    Returns file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    embedder = SentenceTransformer(embed_model)

    all_chunks: List[str] = []
    metadata: List[Dict] = []

    for fp in iter_text_files(input_dir):
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        base = os.path.basename(fp)
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            metadata.append({"source_file": base, "chunk_id": i})

    if not all_chunks:
        raise RuntimeError(f"No .txt files found in {input_dir} or files were empty.")

    # Embed and index
    emb = embedder.encode(all_chunks, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    emb = emb.astype("float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine-like if embeddings are normalized; MiniLM is typically ok.
    # Optional: normalize for true cosine similarity
    faiss.normalize_L2(emb)
    index.add(emb)

    chunks_path = os.path.join(out_dir, "all_chunks.pkl")
    meta_path = os.path.join(out_dir, "chunk_metadata.pkl")
    index_path = os.path.join(out_dir, "rag_index.faiss")

    with open(chunks_path, "wb") as f:
        pickle.dump(all_chunks, f)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    faiss.write_index(index, index_path)

    print(f"Wrote {len(all_chunks)} chunks")
    print(f"Chunks:  {chunks_path}")
    print(f"Meta:    {meta_path}")
    print(f"Index:   {index_path}")
    return chunks_path, meta_path, index_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FAISS RAG index for a directory of .txt book files.")
    ap.add_argument("--input_dir", required=True, help="Directory of .txt files")
    ap.add_argument("--out_dir", required=True, help="Output directory for rag artifacts")
    ap.add_argument("--embed_model", default="all-MiniLM-L6-v2", help="SentenceTransformer model id")
    # IMPORTANT: match defaults with book_format_UPDATED.py so training/inference see similar chunk shapes
    ap.add_argument("--chunk_size", type=int, default=2000, help="Chunk size in characters")
    ap.add_argument("--overlap", type=int, default=200, help="Overlap in characters")
    args = ap.parse_args()

    build_rag_index(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        embed_model=args.embed_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
