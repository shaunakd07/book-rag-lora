"""
book_format.py

Generate RAG-aligned chat fine-tuning data (JSONL) from a directory of .txt book files.

Key idea:
- Each training example includes the retrieved/grounding passage ("CONTEXT") inside the user message.
- The assistant answer must be fully supported by the context, or abstain with:
  "I don't know based on the provided context."

This aligns LoRA training with a RAG inference-time prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Optional

from openai import OpenAI


ABSTAIN = "I don't know based on the provided context."


@dataclass
class Chunk:
    source_file: str
    chunk_id: int
    text: str


def iter_text_files(input_dir: str) -> Iterable[str]:
    for name in sorted(os.listdir(input_dir)):
        if name.lower().endswith(".txt"):
            yield os.path.join(input_dir, name)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into chunks of size `chunk_size` with character overlap.
    Overlap helps keep local coherence and improves QA quality.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[str] = []
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


def _strip_markdown_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # Remove first fence line
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1 :]
        # Remove trailing fence if present
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def generate_qa_from_chunk(
    client: OpenAI,
    model: str,
    chunk: Chunk,
    max_qas: int,
    temperature: float,
    max_retries: int,
    retry_sleep_s: float,
) -> List[Dict[str, Any]]:
    """
    Returns a list of objects: {"question": str, "answer": str, "answerable": bool}
    """
    system = (
        "You create training data for a retrieval-grounded chatbot.\n"
        "Given a CONTEXT passage, write a JSON array (no markdown fences) of up to "
        f"{max_qas} objects with keys: question, answer, answerable.\n\n"
        "Rules:\n"
        "1) Every question must be answerable ONLY from the provided CONTEXT. Ask a mix of straightforward fact-based questions and complex, deep questions that require examples and details to answer.\n"
        "2)Do not ask any questions about ISBN numbers or publishing information. \n"
        "3) If a question is NOT answerable from the CONTEXT, set answerable=false and "
        f'answer must be exactly: "{ABSTAIN}".\n'
        "4) Do not use outside knowledge. Do not invent details. Include 2-5 relevant details from context.\n"
        "5) Return ONLY valid JSON.\n"
    )

    user = f"CONTEXT:\n{chunk.text}"

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            raw = resp.choices[0].message.content or ""
            raw = _strip_markdown_fences(raw)

            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("Model did not return a JSON array.")

            cleaned: List[Dict[str, Any]] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                q = str(item.get("question", "")).strip()
                a = str(item.get("answer", "")).strip()
                answerable = bool(item.get("answerable", True))
                if not q:
                    continue
                if not answerable:
                    a = ABSTAIN
                if not a:
                    # If answer missing, force abstention to avoid garbage training examples
                    a = ABSTAIN
                    answerable = False
                cleaned.append({"question": q, "answer": a, "answerable": answerable})

            return cleaned

        except Exception as e:
            last_err = e
            # Simple backoff
            time.sleep(retry_sleep_s * attempt)

    print(
        f"[WARN] QA generation failed after {max_retries} attempts for "
        f"{os.path.basename(chunk.source_file)} chunk {chunk.chunk_id}: {last_err}"
    )
    return []


def to_chat_example(context: str, question: str, answer: str) -> Dict[str, Any]:
    """
    Produce one JSONL line in chat fine-tuning format that is RAG-aligned:
    """
    system = (
        "You are a helpful assistant answering questions about a book.\n"
        "Answer in 1–3 paragraphs.\n"
        "Include 2–5 specific details from the context.\n"
        "You must ONLY use the provided CONTEXT FROM BOOK.\n"
        f'If the answer is not in the context, say: "{ABSTAIN}".'
    )
    user = f"CONTEXT FROM BOOK:\n{context}\n\nQUESTION:\n{question}"
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create RAG-aligned JSONL chat training data from .txt files.")
    parser.add_argument("--input_dir", required=True, help="Directory containing .txt files.")
    parser.add_argument("--output_file", required=True, help="Path to write JSONL training data.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model used to generate QA pairs.")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=200, help="Chunk overlap in characters.")
    parser.add_argument("--max_qas_per_chunk", type=int, default=8, help="Max QA objects to generate per chunk.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for QA generation.")
    parser.add_argument("--max_retries", type=int, default=4, help="Retries for QA generation calls.")
    parser.add_argument("--retry_sleep_s", type=float, default=1.0, help="Base sleep seconds for retry backoff.")
    parser.add_argument("--include_metadata", action="store_true", help="Include source metadata on each JSONL line.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in your environment.")

    client = OpenAI(api_key=api_key)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    total_written = 0
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for file_path in iter_text_files(args.input_dir):
            with open(file_path, "r", encoding="utf-8") as f_in:
                text = f_in.read()

            chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
            for chunk_id, chunk_str in enumerate(chunks):
                chunk = Chunk(source_file=file_path, chunk_id=chunk_id, text=chunk_str)
                qa_pairs = generate_qa_from_chunk(
                    client=client,
                    model=args.model,
                    chunk=chunk,
                    max_qas=args.max_qas_per_chunk,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    retry_sleep_s=args.retry_sleep_s,
                )

                for qa in qa_pairs:
                    ex = to_chat_example(
                        context=chunk.text,
                        question=qa["question"],
                        answer=qa["answer"],
                    )
                    if args.include_metadata:
                        ex["metadata"] = {
                            "source_file": os.path.basename(chunk.source_file),
                            "chunk_id": chunk.chunk_id,
                            "answerable": qa["answerable"],
                        }
                    f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    total_written += 1

    print(f"Done. Wrote {total_written} training examples to: {args.output_file}")


if __name__ == "__main__":
    main()
