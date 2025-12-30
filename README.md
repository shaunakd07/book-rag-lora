README

Overview

This project fine-tunes a chat LLM (via LoRA) to answer questions about books strictly using retrieved text passages. The pipeline is “RAG-aligned”: the same prompt structure is used during training and inference, where the user message contains a CONTEXT FROM BOOK: block plus the QUESTION:.

1) Input books

Books are provided as plain text files (.txt) in a directory (one or more files).

2) Create RAG-aligned training data (book_format.py)

book_format.py reads the .txt files and splits each book into overlapping character chunks (--chunk_size, --overlap). For each chunk, it calls an LLM to generate multiple question/answer pairs. Each training example is stored in JSONL chat format:

System message: instructs the model to only use provided context and to abstain when unsupported.

User message: contains CONTEXT FROM BOOK:\n{chunk_text}\n\nQUESTION:\n{question}.

Assistant message: contains the answer, or the exact abstention string:
I don't know based on the provided context.

In this stage, “context” = exactly one chunk of the book.

3) Build the retrieval index (book_rag.py)

book_rag.py chunks the same .txt files (same chunking scheme), embeds each chunk using a SentenceTransformer model (default all-MiniLM-L6-v2), normalizes embeddings, and builds a FAISS inner-product index (IndexFlatIP) for cosine-like similarity search. It saves three artifacts:

all_chunks.pkl (raw chunk text)

chunk_metadata.pkl (source file + chunk id)

rag_index.faiss (FAISS vector index)

4) LoRA fine-tuning (book_train.py)

book_train.py loads the JSONL training set and converts each example’s messages into a single model input string via the tokenizer’s chat template. It then masks labels so that loss is computed only on the assistant answer tokens (everything before the assistant answer is set to -100). This trains the model to generate answers conditioned on the system + user message, where the user message contains the context passage.

The script wraps the base model with LoRA (typically targeting attention projection modules) and saves the resulting adapter to disk.

5) RAG + LoRA inference (book_inference.py)

At inference time, book_inference.py loads:

the RAG artifacts (chunks/metadata/index + embedder),

the base model,

the LoRA adapter.

When the user asks a question q, the code:

embeds q,

searches the FAISS index for top-k relevant chunks,

concatenates those chunks into a single context string,

builds a chat prompt with the same training structure (CONTEXT FROM BOOK + QUESTION),

calls model.generate(...) to produce the answer.

In this stage, “context” = the concatenation of top-k retrieved chunks, and the LLM never performs retrieval; it only answers based on the context provided in the prompt.
```mermaid
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
```
```mermaid
sequenceDiagram
  autonumber
  participant U as User
  participant PY as book_inference.py (your code)
  participant ST as SentenceTransformer embedder
  participant F as FAISS index
  participant LLM as Base model + LoRA adapter

  U->>PY: Ask question q
  PY->>ST: encode(q) -> q_emb
  PY->>F: search(q_emb, k) -> top-k ids
  F-->>PY: ids + scores
  PY->>PY: context = join(chunks[ids], "\\n\\n---\\n\\n")
  PY->>LLM: prompt = SYSTEM + USER(CONTEXT + QUESTION)
  LLM-->>PY: answer text (grounded or ABSTAIN)
  PY-->>U: return answer
```

