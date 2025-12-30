README

TXT BOOKS
   ↓
book_format.py
   ↓
formatted_book_data.jsonl   (CONTEXT + QUESTION → ANSWER)
   ↓
book_train.py
   ↓
LoRA adapter

*************************

TXT BOOKS
   ↓
book_rag.py
   ↓
FAISS index + chunk store
   ↓
book_inference.py
   ↓
Answers

Question
  ↓
RAG retrieves chunks  ← (this supplies knowledge)
  ↓
Prompt (context + question)
  ↓
Base model + LoRA  ← (this supplies behavior)
  ↓
Answer


flowchart TB

  %% 1) SOURCE BOOKS
  subgraph S["Source books"]
    B1["Books as .txt files<br/>(one file per book or section)"]
  end

  %% 2) FORMATTING: TRAINING DATA
  subgraph F["Training data formatting (book_format.py)"]
    F0["Load .txt files from --input_dir"]
    F1["Chunk text<br/>chunk_size + overlap"]
    F2["Generate QA pairs per chunk<br/>(OpenAI call)"]
    F3["Validate QA<br/>Unanswerable -> ABSTAIN"]
    F4["Write JSONL chat examples"]
  end

  %% CONTEXT IN TRAINING
  subgraph FC["Context inside training examples"]
    C1["Context = ONE chunk"]
    C2["User message:<br/>CONTEXT FROM BOOK + QUESTION"]
    C3["System message:<br/>Use context only"]
    C4["Assistant:<br/>Answer or ABSTAIN"]
  end

  %% 3) RAG BUILD
  subgraph R["RAG index build (book_rag.py)"]
    R0["Load same .txt files"]
    R1["Chunk text"]
    R2["Embed chunks (SentenceTransformer)"]
    R3["Normalize embeddings"]
    R4["Build FAISS IndexFlatIP"]
    R5["Save index + chunks + metadata"]
  end

  %% 4) TRAINING
  subgraph T["LoRA fine-tuning (book_train.py)"]
    T0["Load JSONL dataset"]
    T1["Apply chat template"]
    T2["Tokenize prompt + answer"]
    T3["Mask labels before assistant"]
    T4["Train LoRA adapter"]
    T5["Save adapter"]
  end

  %% 5) INFERENCE
  subgraph I["Inference with RAG (book_inference.py)"]
    I0["Load FAISS index + chunks"]
    I1["Load base model + LoRA"]
    I2["Embed user question"]
    I3["Retrieve top-k chunks"]
    I4["Concatenate chunks as context"]
    I5["Build prompt (same as training)"]
    I6["Generate answer"]
  end

  %% CONNECTIONS
  B1 --> F0
  F0 --> F1
  F1 --> F2
  F2 --> F3
  F3 --> F4
  
  %% Link training data logic to context format
  F1 --> C1
  C1 --> C2
  C2 --> C3
  C3 --> C4
  C4 --> F4

  B1 --> R0
  R0 --> R1
  R1 --> R2
  R2 --> R3
  R3 --> R4
  R4 --> R5

  F4 --> T0
  T0 --> T1
  T1 --> T2
  T2 --> T3
  T3 --> T4
  T4 --> T5

  R5 --> I0
  I0 --> I1
  I1 --> I2
  I2 --> I3
  I3 --> I4
  I4 --> I5
  I5 --> I6


sequenceDiagram
  autonumber
  participant U as User
  participant PY as book_inference.py (your code)
  participant ST as SentenceTransformer (embedder)
  participant F as FAISS index
  participant LLM as Base model + LoRA adapter

  U->>PY: Ask question q
  PY->>ST: encode(q) -> q_emb
  PY->>F: search(q_emb, k) -> top-k ids
  F-->>PY: ids + scores
  PY->>PY: context = join(chunks[ids], "\n\n---\n\n")
  PY->>LLM: prompt = SYSTEM + USER(CONTEXT FROM BOOK + QUESTION)\nGenerate(...)
  LLM-->>PY: answer text (grounded or ABSTAIN)
  PY-->>U: print answer 
