import os
import argparse
import pickle
import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


try:
    from peft import PeftModel
except Exception:
    PeftModel = None


ABSTAIN = "I don't know based on the provided context."


def load_rag(rag_dir: str, embed_model: str):
    chunks_path = os.path.join(rag_dir, "all_chunks.pkl")
    meta_path = os.path.join(rag_dir, "chunk_metadata.pkl")
    index_path = os.path.join(rag_dir, "rag_index.faiss")

    with open(chunks_path, "rb") as f:
        all_chunks = pickle.load(f)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    index = faiss.read_index(index_path)
    embedder = SentenceTransformer(embed_model)
    return all_chunks, metadata, index, embedder


def retrieve(all_chunks, metadata, index, embedder, query: str, k: int):
    q = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    distances, indices = index.search(q, k)

    hits = []
    for score, idx in zip(distances[0].tolist(), indices[0].tolist()):
        if idx < 0:
            continue
        hits.append({"score": score, "text": all_chunks[idx], "meta": metadata[idx]})
    return hits


def build_prompt(tokenizer, context: str, question: str) -> str:
    system = (
        "You are a helpful assistant answering questions about a book.\n"
        "You must ONLY use the provided CONTEXT FROM BOOK.\n"
        f'If the answer is not in the context, say: "{ABSTAIN}". \n'
        "When the answer is in the context, write a thorough answer: "
        "explain reasoning, include relevant details from the context, and "
        "use 1–3 paragraphs unless the question is purely factual."
    )
    user = (
        f"CONTEXT FROM BOOK:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer in 1–3 paragraphs.\n"
        "- Include 2–5 specific details from the context.\n"
        "- If helpful, add a short 'Why this matters' paragraph.\n"
    )
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt

def main() -> None:
    ap = argparse.ArgumentParser(description="RAG + LoRA inference for book QA.")
    ap.add_argument("--rag_dir", required=True, help="Directory containing all_chunks.pkl, chunk_metadata.pkl, rag_index.faiss")
    ap.add_argument("--embed_model", default="all-MiniLM-L6-v2", help="SentenceTransformer embedder")
    ap.add_argument("--base_model", default='meta-llama/Llama-3.2-1B-Instruct', help="Base HF model id or local path (same as training base)")
    ap.add_argument("--lora_path", required=True, help="Path to LoRA adapter directory")
    ap.add_argument("--k", type=int, default=8, help="Top-k chunks to retrieve")
    ap.add_argument("--max_new_tokens", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_chunks, metadata, index, embedder = load_rag(args.rag_dir, args.embed_model)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if args.lora_path:
        if PeftModel is None:
            raise RuntimeError("peft is not installed, but --lora_path was provided.")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model.eval()

    print("Ready. Type questions; Ctrl+C to exit.\n")
    while True:
        q = input("You: ").strip()
        if not q:
            continue

        hits = retrieve(all_chunks, metadata, index, embedder, q, args.k)
        context = "\n\n---\n\n".join(
            [h["text"] for h in hits]
        )

        prompt = build_prompt(tokenizer, context=context, question=q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=150,
                temperature=args.temperature,
                do_sample=True,
                repetition_penalty=1.15,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen = out[0][inputs["input_ids"].shape[-1]:]
        ans = tokenizer.decode(gen, skip_special_tokens=True).strip()

        # Optional: print citations
        print("\nAssistant:", ans)
        print("\nSources:")
        for i, h in enumerate(hits, 1):
            m = h["meta"]
            print(f"  [{i}] {m.get('source_file')} chunk {m.get('chunk_id')} (score={h['score']:.3f})")
        print()


if __name__ == "__main__":
    main()
