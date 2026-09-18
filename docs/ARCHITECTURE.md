# SovereignAI — Architecture (Phase 0)

**Problem Statement:** SIH 117 — a self-hosted, air-gapped AI workbench
for refineries / PSUs / defence-linked manufacturing units.

**Author:** Akash Upadhyay (B.Tech CS/AI-ML)

**Status:** Living document, updated every phase. The authoritative
phase-by-phase status lives in `docs/BUILD_STATUS.md`.

---

## 1. One-paragraph summary

SovereignAI is a local application that lets an organization's staff
work with an AI assistant — chat, document analysis, coding, scanned-
document understanding, spreadsheet/word/slide generation — **without
any data ever leaving the machine it runs on**. It is built around a
from-scratch, self-trained educational language model, a pluggable
multi-model router, an agent loop with local tools, local retrieval-
augmented generation, local OCR/vision, and a Flask-based local web
UI — backed by SQLite and enforced by an explicit offline/security
layer.

---

## 2. Honesty ledger (kept accurate every phase)

| Component | What it actually is |
|---|---|
| Custom MiniLLM (tokenizer, Transformer, training loop) | **Genuinely built from scratch** in this repo, using PyTorch only for tensors/autograd/CUDA — not a pretrained model. Small (target ~4M-20M parameters). Educational/research grade, explicitly **not** a frontier model. |
| Additional local models (Phase 8+) | Open-weight models run locally through an adapter interface. Never a cloud API. |
| Task router (Phase 9) | **Rule-based** unless/until a learned classifier is explicitly built and labelled as such. |
| OCR (Phase 13) | Local text extraction from scans — **not** "vision understanding" unless a vision model is separately documented. |
| Vision (Phase 14) | Local vision model inference, scoped to exactly what it demonstrably does. |
| Sandboxed execution (Phase 15) | Isolated local subprocess execution with restrictions — not a hardened production-grade VM/container sandbox. |
| "Air-gapped" claim | Backed by a network monitor (Phase 18) producing **visible evidence** of zero outbound calls — not just an assertion. |

This table is expanded into `docs/FINAL_AUDIT.md` (Phase 24) once every
phase is complete.

---

## 3. Comprehensive System Map & Architecture Blueprint

```mermaid
flowchart TB
    %% Subgraphs
    subgraph UI["🖥️ CLIENT & PRESENTATION TIER (Phase 20)"]
        direction TB
        ChatUI["💬 Chat Workspace<br/>• Streaming inference<br/>• Model & mode selector<br/>• Stop generation control"]
        DocsUI["📄 Document Explorer<br/>• Multi-file upload<br/>• PDF / DOCX / TXT / MD<br/>• Chunk visualizer"]
        KBUI["🧠 Knowledge Catalog<br/>• FineWeb-Edu 1.6B index<br/>• Hybrid semantic retrieval<br/>• Vector store status"]
        ToolsUI["⚙️ Tool Runner Studio<br/>• AST Math calculator<br/>• Scanned OCR extraction<br/>• Office artifact generator"]
        SecUI["🔒 Air-Gap Security Monitor<br/>• 0-socket egress indicator<br/>• Real-time connection log<br/>• Tamper-evident audit trail"]
    end

    subgraph Gateway["⚡ APPLICATION GATEWAY & SECURITY LAYER (Phase 18-19)"]
        direction TB
        API["🌐 Local Flask REST Gateway<br/>(127.0.0.1:5000 · Zero WAN / Strict Localhost)"]
        NetMon["🛡️ Socket Interceptor & Network Monitor<br/>• Monkey-patched socket.socket<br/>• Blocks outbound internet traffic<br/>• Real-time socket connection logging"]
        RBAC["👥 Role-Based Access Controller<br/>• Admin / Engineer / Auditor roles<br/>• Strict capability & tool bounds"]
    end

    subgraph Routing["🧭 TASK ROUTING & DISPATCH (Phase 9)"]
        direction TB
        Classifier["🎯 Intent Classifier<br/>• Regex & pattern matching<br/>• Domain keywords<br/>• Fallback arbitration"]
        Policy["📋 Dispatch Policies<br/>• Direct LLM synthesis<br/>• Grounded RAG query<br/>• Structured SQL inquiry<br/>• Deterministic tool run<br/>• Multi-step agent workflow"]
    end

    subgraph AgentTriad["🤖 AUTONOMOUS AGENT TRIAD (Phase 11)"]
        direction TB
        Planner["📋 Task Planner<br/>• Decomposes goal into DAG<br/>• Assigns deterministic tools<br/>• Establishes halt constraints"]
        Executor["⚡ Step Executor<br/>• Dispatches tool actions<br/>• Traverses execution plan<br/>• Handles retries & fallbacks"]
        Verifier["✅ State Verifier<br/>• Tests schema & correctness<br/>• Verifies safety invariants<br/>• Prevents hallucinated leaps"]
        Scratchpad["🧠 Memory & Scratchpad<br/>• Ephemeral state history<br/>• Serializable task context"]
    end

    subgraph Models["🧠 MULTI-MODEL INFERENCE SUBSYSTEM (Phase 3-8)"]
        direction TB
        MiniLLM["🧬 Custom MiniLLM (~10.2M)<br/>• From scratch in PyTorch<br/>• RoPE + SwiGLU + RMSNorm<br/>• Tied embedding weights<br/>• Autoregressive sampling"]
        BPETok["🔤 Byte-Level BPE Tokenizer<br/>• 8,192-token vocabulary<br/>• Written from raw bytes<br/>• Zero external binary blobs"]
        IndModel["🏭 Industrial 80M Adapter<br/>• PSU & refinery reasoning<br/>• Edge GPU accelerated<br/>• Fully local open-weight"]
        EmbModel["📐 MiniLM Adapter<br/>• 384-d dense vectorizer<br/>• Sentence-level semantics<br/>• Offline local inference"]
        VisionModel["👁️ Local Vision / OCR Adapter<br/>• Multimodal schematics<br/>• Engineering blueprint OCR<br/>• Tesseract integration"]
    end

    subgraph ToolsRegistry["🛠️ DETERMINISTIC LOCAL TOOL REGISTRY (Phase 10, 13-16)"]
        direction TB
        ASTCalc["🧮 Safe AST Calculator<br/>• Zero eval() vulnerabilities<br/>• Math & engineering AST"]
        FSTool["📁 Local Filesystem Tool<br/>• Sandboxed read / write<br/>• Strict directory boundary"]
        OCRTool["🔍 PDF / OCR Extractor<br/>• PyMuPDF text parser<br/>• Tesseract OCR engine"]
        SheetTool["📊 Spreadsheet Processor<br/>• OpenPyXL CSV/XLSX<br/>• Tabular aggregation"]
        OfficeDoc["📝 Office Deliverable Writer<br/>• Word (.docx) generator<br/>• PowerPoint (.pptx) slides<br/>• Excel (.xlsx) workbooks"]
        Sandbox["📦 Subprocess Code Sandbox<br/>• Restricted execution<br/>• Execution timeout guard"]
    end

    subgraph RAGEngine["📚 HYBRID RAG & KNOWLEDGE ENGINE (Phase 12)"]
        direction TB
        Ingest["📥 Document Ingestion<br/>• Multi-format ingestion<br/>• PDF, DOCX, TXT, MD, JSONL"]
        Chunker["✂️ Sliding-Window Chunker<br/>• 256-512 token windows<br/>• Configurable chunk overlap"]
        VecIndex["⚡ Dense Vector Index<br/>• Cosine similarity search<br/>• Fast vector math in NumPy/Torch"]
        FineWeb["🌐 FineWeb-Edu 1.6B Corpus<br/>• Pre-indexed knowledge store<br/>• Fast local semantic lookup"]
        Citations["📑 Citation Grounding Engine<br/>• Document & page metadata<br/>• Paragraph-level attribution"]
    end

    subgraph Storage["💾 PERSISTENCE & RELATIONAL STORAGE (Phase 17)"]
        direction TB
        SQLiteDB[("🗄️ SQLite Database (sovereign_ai.db)<br/>• users & sessions<br/>• conversations & messages<br/>• documents & chunks<br/>• task_plans & step_history<br/>• append-only audit_log")]
        VectorStore[("💾 Vector Storage Cache<br/>• .npy / .bin vector arrays<br/>• FineWeb embeddings")]
    end

    %% Flow connections
    ChatUI --> API
    DocsUI --> API
    KBUI --> API
    ToolsUI --> API
    SecUI --> API

    API --> NetMon
    API --> RBAC
    API --> Classifier
    Classifier --> Policy

    Policy -->|Direct Synthesis| MiniLLM
    Policy -->|Domain Synthesis| IndModel
    Policy -->|Grounded Q&A| Ingest
    Policy -->|Structured Query| SQLiteDB
    Policy -->|Tool Invocation| ToolsRegistry
    Policy -->|Multi-Step Goal| Planner

    Planner --> Executor
    Executor --> ToolsRegistry
    Executor --> Models
    Executor --> Ingest
    Executor --> Scratchpad
    Scratchpad --> Verifier
    Executor --> Verifier
    Verifier -->|Pass| API
    Verifier -->|Retry| Executor

    MiniLLM --- BPETok
    IndModel --- BPETok

    Ingest --> Chunker
    Chunker --> EmbModel
    EmbModel --> VecIndex
    VecIndex --> Citations
    FineWeb --> VecIndex
    Citations --> MiniLLM

    ASTCalc --> Scratchpad
    FSTool --> Scratchpad
    OCRTool --> Scratchpad
    SheetTool --> Scratchpad
    OfficeDoc --> Scratchpad
    Sandbox --> Scratchpad

    VecIndex --> VectorStore
    Ingest --> SQLiteDB
    API --> SQLiteDB
    NetMon -.->|Socket Audit Records| SQLiteDB
    RBAC -.->|Permission Audit| SQLiteDB

    %% Styling
    classDef ui fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef gateway fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef router fill:#311042,stroke:#c084fc,stroke-width:2px,color:#f8fafc;
    classDef agent fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef models fill:#4a044e,stroke:#f472b6,stroke-width:2px,color:#f8fafc;
    classDef tools fill:#431407,stroke:#fb923c,stroke-width:2px,color:#f8fafc;
    classDef rag fill:#022c22,stroke:#2dd4bf,stroke-width:2px,color:#f8fafc;
    classDef storage fill:#172554,stroke:#60a5fa,stroke-width:2px,color:#f8fafc;

    class ChatUI,DocsUI,KBUI,ToolsUI,SecUI ui;
    class API,NetMon,RBAC gateway;
    class Classifier,Policy router;
    class Planner,Executor,Verifier,Scratchpad agent;
    class MiniLLM,BPETok,IndModel,EmbModel,VisionModel models;
    class ASTCalc,FSTool,OCRTool,SheetTool,OfficeDoc,Sandbox tools;
    class Ingest,Chunker,VecIndex,FineWeb,Citations rag;
    class SQLiteDB,VectorStore storage;
```

### Subsystem Responsibility & Layer Mapping

| Subsystem Layer | Directory | Core Components | Operational Responsibility |
|---|---|---|---|
| **Client UI** | `app/frontend/`<br/>`app/frontend-src/` | `index.html`, React/Vite App, `chatStore.ts` | 7-panel operator console: streaming chat, document upload, RAG citation inspector, offline audit monitor. |
| **Gateway & Security** | `app/backend/`<br/>`security/` | `app.py`, `network_monitor.py`, `permissions.py` | Local Flask REST server (127.0.0.1:5000), socket interceptor with 0% outbound network egress, and RBAC controller. |
| **Task Router** | `router/` | `task_classifier.py`, `policies.py`, `router.py` | Deterministic intent classifier dispatching queries to LLM, RAG, SQL, Tools, or Agent Triad. |
| **Agent Triad** | `agents/` | `planner.py`, `executor.py`, `verifier.py`, `state.py` | Goal decomposition DAG, tool execution engine, invariant verifier, and serializable task memory. |
| **Model Layer** | `models/`<br/>`tokenizer/` | `models/custom_minilm/`, `models/adapters/`, `tokenizer/` | Custom ~10.2M Transformer, Byte-level BPE tokenizer (8,192 vocab), Industrial 80M adapter, MiniLM embedding adapter. |
| **Deterministic Tools**| `tools/` | `calculator.py`, `filesystem.py`, `ocr.py`, `word.py`, `powerpoint.py`, `code_sandbox.py` | Math AST calculator (eval-free), sandboxed filesystem I/O, Tesseract OCR, DOCX/PPTX/XLSX writers, isolated sandbox. |
| **Hybrid RAG** | `rag/` | `ingest.py`, `chunking.py`, `embeddings.py`, `fineweb.py`, `citations.py` | Local multi-format ingestion, sliding-window chunking, dense vector index, FineWeb-Edu 1.6B knowledge store, and citation engine. |
| **Persistence** | `database/` | `schema.sql`, `db.py`, `repositories/conversations.py` | SQLite schema and repository storing users, conversations, document chunks, task steps, and append-only audit log. |
| **Air-Gap Security**| `security/` | `offline_mode.py`, `audit.py` | Socket-level egress blocker, automated air-gap verification script, and immutable hash-verified audit log. |

## 4. Data flow (training-time)

```mermaid
flowchart LR
    Raw[data/raw] --> Clean[data/cleaned]
    Clean --> Filter[filter by length]
    Filter --> Split[train / val split]
    Split --> Processed[data/processed]
    Processed --> Tokenize[BPE tokenizer]
    Tokenize --> TrainLoop[Training loop]
    TrainLoop --> Checkpoint[(checkpoints/)]
```

## 5. Inference / agent flow (runtime)

```mermaid
flowchart LR
    Request[User request] --> Router
    Router -->|coding| CodeModel[Coding-capable local model]
    Router -->|doc / summary| MiniLLM
    Router -->|image / scan| Vision
    Router -->|retrieval| RAG
    CodeModel --> Agent
    MiniLLM --> Agent
    Vision --> Agent
    RAG --> Agent
    Agent --> Tools
    Tools --> Verifier
    Verifier -->|ok| Deliverable[DOCX / XLSX / PPTX / answer]
    Verifier -->|fail| Agent
```

## 6. Security / air-gap flow

```mermaid
flowchart TB
    App[Application process] --> NetMon[Network Monitor]
    NetMon -->|logs every socket attempt| AuditLog[(audit log)]
    NetMon -->|blocks / flags| Alert[Visible alert in UI + logs]
    AuditLog --> Proof[docs/OFFLINE_PROOF.md evidence]
```

---

## 7. Folder structure

Two intentional additions beyond the originally sketched structure,
both justified:

| Addition | Reason |
|---|---|
| `core/` | Shared config + logging + corpus-reading utilities used by every phase. Avoids duplicating path/logging logic across 24 phases. |
| `tokenizer/tests/` | Standalone tokenizer test suite, as required. |

Directories for phases not yet built (`training/`, `router/`,
`agents/`, `tools/`, `rag/`, `database/`, `security/`, `app/`,
`models/custom_minilm/`, `models/adapters/`, `models/vision/`) exist
as **empty, reserved folders** with only a short `README.md` — no
placeholder code, no `pass` stubs. They are populated only when their
phase is reached.

## 8. Dependency plan

| Package | Used for | Phase | Runtime/dev | Internet at runtime? |
|---|---|---|---|---|
| torch | tensors, autograd, CUDA | 1+ | runtime | No |
| numpy | numeric utilities | 1+ | runtime | No |
| pytest | test running | 1+ | dev | No |
| flask *(later)* | local backend (not FastAPI) | 19 | runtime | No |
| python-docx *(later)* | Word generation | 16 | runtime | No |
| python-pptx *(later)* | PowerPoint generation | 16 | runtime | No |
| openpyxl *(later)* | Excel generation | 16 | runtime | No |
| pymupdf *(later)* | PDF parsing/rendering | 13 | runtime | No |
| pytesseract *(later)* | OCR (needs local Tesseract binary) | 13 | runtime | No (after install) |
| faiss-cpu *(later)* | local vector index | 12 | runtime | No |

Every license/purpose is re-confirmed in `docs/FINAL_AUDIT.md` once
its phase is reached.

## 9. Model strategy

1. **Phases 3-7** — build one tiny (~4-20M parameter) decoder-only
   Transformer completely from scratch (own tokenizer, own attention,
   own RMSNorm, own RoPE, own SwiGLU, own training loop), sized for a
   6 GB VRAM RTX 4050 laptop.
2. **Phase 8** — wrap it behind a `ModelProvider` interface so it is
   just *one* of potentially several registered models.
3. **Phase 8+ (optional, hardware-permitting)** — register one
   additional open-weight local model purely through the adapter
   interface, with no changes to earlier phases. License / VRAM /
   offline-compatibility verified before adding it.

## 10. Data strategy

- Only synthetic (self-authored) or clearly-licensed public data.
- Every raw file gets a metadata JSON: `source`, `license`,
  `processing_notes`, `sha256`.
- Raw → cleaned → processed → tokenized, each stage on disk and
  inspectable.
- A tiny synthetic corpus proves the **pipeline** works — it is not
  evidence of model fluency.

## 11. 24-phase roadmap

| # | Phase | One-line goal |
|---|---|---|
| 0 | Architecture | This document |
| 1 | Foundation | Repo skeleton, config, logging, GPU check |
| 2 | Data | raw→cleaned→processed pipeline + synthetic corpus |
| 3 | Tokenizer | From-scratch byte-level BPE + tests |
| 4 | NN foundations | Embeddings, RMSNorm, RoPE, attention math (unit tested) |
| 5 | Transformer/MiniLLM | Assemble the full model |
| 6 | Training | Real autoregressive training loop |
| 7 | Checkpoint/Inference | Save/load/resume + generation |
| 8 | Multi-model abstraction | ModelProvider / adapters |
| 9 | Router | Rule-based task → model/tool routing |
| 10 | Tools | Filesystem, calculator, PDF, sandbox, etc. |
| 11 | Agent | Planner/executor/verifier/memory loop |
| 12 | RAG | Local ingestion/embedding/retrieval |
| 13 | OCR | Scanned-PDF text extraction |
| 14 | Vision | Local image understanding |
| 15 | Sandboxed coding | Isolated code execution + verification |
| 16 | Document outputs | DOCX/XLSX/PPTX generation |
| 17 | SQLite | Application persistence |
| 18 | Security | Offline enforcement, audit, permissions |
| 19 | Backend | Flask app tying subsystems together |
| 20 | UI | Local web workbench |
| 21 | End-to-end | Full integration |
| 22 | Evaluation | Benchmarks/eval scripts |
| 23 | SIH demo | Demos A-E |
| 24 | Docs/audit | Final documentation + `FINAL_AUDIT.md` |

Live status of each row: `docs/BUILD_STATUS.md`.

## 12. SIH demo scenarios (built in Phase 23)

- **Demo A** — Scanned inspection report → OCR → findings → agent plan → DOCX approval note.
- **Demo B** — Coding request → router → coding model → sandbox → verified code.
- **Demo C** — Local knowledge question → RAG retrieval + citations → answer.
- **Demo D** — Multimodal document → vision/OCR → analysis → deliverable.
- **Demo E** — Sovereignty proof → network monitor shows zero external calls.

## 13. Risks & limitations (kept honest)

- A 4-20M parameter model will **not** produce fluent, general-purpose
  text — it is a learning vehicle for how LLMs actually work, not a
  chat product. The workbench's practical usefulness comes from
  routing real tasks to a stronger open-weight local model, while the
  from-scratch model demonstrates genuine understanding of the
  internals (the thing judges will actually ask about).
- 6 GB VRAM limits batch size / context length; staged experiments
  (10-50 steps → 100-500 steps → longer run) exist specifically to
  make this constraint visible rather than hidden.
- OCR/vision quality depends entirely on whichever local model is
  chosen in Phase 13/14 — will be scoped honestly, not oversold.
- The code sandbox (Phase 15) reduces risk but is not a substitute for
  a hardened, audited container runtime; called out explicitly
  wherever the sandbox is documented.

## 14. Definition of "done" (whole project)

1. All 24 phases have an entry in `docs/BUILD_STATUS.md` marked
   COMPLETE with a passing test command.
2. Demos A-E (§12) run end-to-end on the Windows 11 / RTX 4050
   development machine.
3. `docs/FINAL_AUDIT.md` accurately answers every question in its
   template (custom vs open-weight vs rule-based, data provenance,
   licenses, what never leaves the machine, limitations).
4. No component silently depends on internet access at runtime.
