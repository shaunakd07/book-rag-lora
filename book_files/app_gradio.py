import pickle
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------- config (match your paths) --------
RAG_DIR = "./rag"          # contains rag_index.faiss + all_chunks.pkl
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LORA_PATH = "./transformer_results/1b/checkpoint-791"
TOP_K_DEFAULT = 4

ABSTAIN = "I don't know based on the provided context."
SYSTEM = (
        "You are a helpful assistant answering questions about a book.\n"
        "You must ONLY use the provided CONTEXT FROM BOOK.\n"
        f'If the answer is not in the context, say: "{ABSTAIN}". \n'
        "When the answer is in the context, write a thorough answer: "
        "explain reasoning, include relevant details from the context, and "
        "use 1–3 paragraphs unless the question is purely factual."
)

# -------- load once on startup --------
index = faiss.read_index(f"{RAG_DIR}/rag_index.faiss")
all_chunks = pickle.load(open(f"{RAG_DIR}/all_chunks.pkl", "rb"))

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, LORA_PATH)
model.eval()

def retrieve_context(q: str, k: int) -> str:
    q_emb = embedder.encode([q], normalize_embeddings=True)
    D, I = index.search(q_emb, k)
    hits = [all_chunks[i] for i in I[0] if i != -1]
    return "\n\n---\n\n".join(hits)

def answer(question: str, k: int, max_new_tokens: int, temperature: float):
    context = retrieve_context(question, k)
    clean_question = question.strip()

    # Use a clear structure for the user message
    user_content = (
        f"CONTEXT FROM BOOK:\n{context}\n\n"
        f"QUESTION:\n{clean_question}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer in 1–3 paragraphs.\n"
        "- Include 2–5 specific details from the context.\n"
        "- If helpful, add a short 'Why this matters' paragraph.\n"
    )
    
    messages = [
        {"role": "system", "content": SYSTEM},
        
        {"role": "user", "content": user_content},
    ]

    # apply_chat_template is the correct way to ensure SYSTEM prompt is used
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    # ---- generate ----
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            do_sample=float(temperature) > 0,
            top_p=0.9,
            repetition_penalty=1.1, 
            pad_token_id=tokenizer.eos_token_id 
        )

    # ---- decode ONLY the assistant answer ----
    answer_ids = output_ids[0][prompt_len:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    return answer_text, context


with gr.Blocks(title="Local Book  Chat") as demo:

    with gr.Row():
        question = gr.Textbox(label="Question", placeholder="Ask something about the book...", lines=2)

    with gr.Row():
        k = gr.Slider(1, 12, value=TOP_K_DEFAULT, step=1, label="Top-k chunks (retrieval)")
        max_new = gr.Slider(32, 1024, value=256, step=32, label="Max new tokens")
        temp = gr.Slider(0.0, 1.2, value=0.2, step=0.05, label="Temperature")

    btn = gr.Button("Ask")

    answer_box = gr.Textbox(label="Answer", lines=10)
    ctx_box = gr.Textbox(label="Retrieved context", lines=12)

    btn.click(
        fn=answer,
        inputs=[question, k, max_new, temp],
        outputs=[answer_box, ctx_box],
    )


demo.launch(server_name="127.0.0.1", server_port=7860)
