"""
book_openai_rag.py

Uses existing local FAISS + SentenceTransformer RAG (built by book_rag.py)
but generate the final answer by calling the OpenAI API (o4-mini by default)
via the Responses API.

Usage example:
  python book_openai_rag.py \
    --rag_dir ./rag \
    --embed_model sentence-transformers/all-MiniLM-L6-v2 \
    --model o4-mini \
    --k 8

Requires:
  pip install openai faiss-cpu sentence-transformers numpy
Set env var:
  OPENAI_API_KEY=...
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI


ABSTAIN = "I don't know based on the provided context."


def load_rag(rag_dir: str, embed_model: str) -> Tuple[List[str], List[Dict[str, Any]], faiss.Index, SentenceTransformer]:
    chunks_path = os.path.join(rag_dir, "all_chunks.pkl")
    meta_path = os.path.join(rag_dir, "chunk_metadata.pkl")
    index_path = os.path.join(rag_dir, "rag_index.faiss")

    with open(chunks_path, "rb") as f:
        all_chunks: List[str] = pickle.load(f)
    with open(meta_path, "rb") as f:
        metadata: List[Dict[str, Any]] = pickle.load(f)

    index = faiss.read_index(index_path)
    embedder = SentenceTransformer(embed_model)
    return all_chunks, metadata, index, embedder


def retrieve(
    all_chunks: List[str],
    metadata: List[Dict[str, Any]],
    index: faiss.Index,
    embedder: SentenceTransformer,
    query: str,
    k: int,
) -> List[Dict[str, Any]]:
    q = embedder.encode([query], convert_to_numpy=True).astype("float32")
    # If your index was built with normalized vectors, normalize query too.
    faiss.normalize_L2(q)
    distances, indices = index.search(q, k)

    hits: List[Dict[str, Any]] = []
    for score, idx in zip(distances[0].tolist(), indices[0].tolist()):
        if idx < 0:
            continue
        hits.append({"score": float(score), "text": all_chunks[idx], "meta": metadata[idx]})
    return hits


def build_messages(context: str, question: str) -> Tuple[str, str]:
    """
    Returns (instructions, input_text) for the Responses API.
    """
    instructions = (
        "You are a helpful assistant answering questions using retrieval-augmented context.\n"
        "You must ONLY use the provided CONTEXT.\n"
        f'If the answer is not in the context, say exactly: "{ABSTAIN}".\n'
        "When the answer IS in the context, write a thorough answer.\n"
        "Include 2–5 specific details from the context.\n"
    )

    input_text = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n"
    )
    return instructions, input_text


def main() -> None:
    ap = argparse.ArgumentParser(description="Local FAISS RAG + OpenAI Responses API generation (o4-mini).")
    ap.add_argument("--rag_dir", required=True, help="Directory with rag_index.faiss, all_chunks.pkl, chunk_metadata.pkl")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer embedder id")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model id (e.g., o4-mini, gpt-4o-mini)")
    ap.add_argument("--k", type=int, default=8, help="Top-k chunks to retrieve")
    ap.add_argument("--max_output_tokens", type=int, default=900, help="Upper bound for output tokens (includes reasoning)")
    ap.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    ap.add_argument("--show_sources", action="store_true", help="Print retrieved sources after each answer")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in your environment.")

    client = OpenAI()

    all_chunks, metadata, index, embedder = load_rag(args.rag_dir, args.embed_model)

    print("Ready. Type questions; Ctrl+C to exit.\n")
    while True:
        q = input("You: ").strip()
        if not q:
            continue

        hits = retrieve(all_chunks, metadata, index, embedder, q, args.k)
        context = "\n\n---\n\n".join([h["text"] for h in hits])
        print("\n[DEBUG] Retrieved context:\n")
        print(context)
        print("\n" + "Retrieved context OVER... answer below:\n"+"=" * 80 + "\n")
        instructions, input_text = build_messages(context=context, question=q)

        # Responses API call
        resp = client.responses.create(
            model=args.model,
            instructions=instructions,
            input=input_text,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            store=False,  # set True if you want OpenAI to store responses for later retrieval
        )

        answer = (resp.output_text or "").strip()
        print("\nAssistant:", answer)

        if args.show_sources:
            print("\nSources:")
            for i, h in enumerate(hits, 1):
                m = h["meta"] or {}
                print(f"  [{i}] {m.get('source_file')} chunk {m.get('chunk_id')} (score={h['score']:.3f})")
        print()


if __name__ == "__main__":
    main()
