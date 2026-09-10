# SovereignAI Workbench — Judge Presentation Guide

> **SIH 2026 | Problem Statement: SIH 117**
> Duration: 5-10 minutes live demo

---

## OPENING STATEMENT (30 seconds)

> *"We built a self-hosted AI workbench that runs 100% on this machine with no internet
> connection. It handles document analysis, knowledge retrieval, code generation,
> and engineering calculations for industrial environments where confidential data
> cannot leave the network. We built the core AI components — the tokenizer,
> the neural network architecture, and the RAG infrastructure — from scratch in Python
> and PyTorch, without using any pre-built Transformers libraries."*

---

## DEMO FLOW (8 minutes)

### 1. Open the workbench (30 sec)
```powershell
python scripts\start_backend.py
# Open browser: http://localhost:5000
```
Point to the **Security panel** → Air-Gap Status: **OFFLINE CONFIRMED ✓**

---

### 2. Scenario 4 — Engineering Calculation (1 min) [most reliable, no Ollama needed]
```
Run in chat:
"Calculate the pressure drop: (0.02 * 50 * 1000 * 0.637 * 0.637) / (2 * 0.1)"
```
- Show: calculator tool fires, result returned
- Say: *"This uses a safe AST-based calculator — zero eval() or exec() — pure math parsing"*

---

### 3. Scenario 3 — Code Generation + Sandbox (2 min) [needs Ollama]
```
Run in Agent tab:
"Write a Python script that calculates the average, max, and min of
[45.2, 47.1, 43.8, 46.5, 48.0, 44.9, 45.7]. Then run it."
```
- Show: model generates code, sandbox executes it, output captured
- Say: *"Code runs in an isolated subprocess — not the main process. Timeout kills infinite loops."*

---

### 4. Scenario 2 — RAG Knowledge Base (2 min)
First ingest a document in the **Knowledge Base** tab:
```
Text: "Compressor seal replacement procedure: isolate the compressor, depressurize the line,
remove the bearing housing cover, extract the old seal, install new SKF 6205 seal,
reassemble with torque 45 Nm, test for leaks before restarting."
Source: compressor_sop.txt
```
Then search:
```
"What is the procedure for replacing a compressor seal?"
```
- Show: cited answer with source name and relevance score
- Say: *"The vector index is pure Python — 30 lines of code. No Pinecone, no Chroma, no cloud."*

---

### 5. Security + Audit (1 min)
- Show **Audit Log** tab: every tool call logged
- Show **Security** tab: network monitor report
- Say: *"Every tool call appears here. Zero external connections. This is the evidence."*

---

### 6. Architecture explanation (1 min)
Point to `docs/HONESTY_LEDGER.md` on screen:
- *"The custom MiniLLM is 5M parameters trained on the demo corpus — it proves we understand
  the math. For real capability, the platform routes to locally-served open-weight models
  like Phi-3 and Mistral via Ollama."*
- *"The honesty ledger documents exactly what's custom vs. pre-trained — we're not claiming
  to have invented the Transformer."*

---

## KEY NUMBERS TO MEMORISE

| Metric | Value |
|--------|-------|
| Custom MiniLLM parameters | ~4.9M |
| Tokenizer vocab size | 600 (BPE) |
| Tokenizer tests passing | 14 |
| Phase 5 (model) tests passing | 17 |
| E2E tests passing | 25 |
| Routing accuracy (labeled set) | ~90%+ |
| External API calls at runtime | **ZERO** |
| Database | Local SQLite |
| Vector DB | Custom Python (no cloud) |
| Lines of custom code (approx.) | 5,000+ |

---

## JUDGE QUESTIONS — PRE-ANSWERED

**"What if I ask it something complex, will it hallucinate?"**
> *"The platform routes complex queries to Ollama/phi3:mini. For RAG answers, it
> shows citations — you can see exactly which document it retrieved the answer from.
> The custom MiniLLM is for demonstrating architecture knowledge, not production chat."*

**"How is this different from just using ChatGPT locally?"**
> *"Three things: (1) we built the neural network architecture ourselves — not using the
> Transformers library. (2) the RAG infrastructure, security layer, and tool system are
> all custom. (3) everything is air-gapped — no API keys, no cloud account, no telemetry."*

**"Can it read PDFs?"**
> *"Yes — via local pdfminer for text-layer PDFs, and local Tesseract OCR for scanned
> image PDFs. No cloud OCR, no Google Vision API."*

**"What would you add with more time?"**
> *"Phase 21 is a larger training run with the real domain corpus. Phases 22-23 add
> a fine-tuned domain model. The infrastructure is already built — it's a training budget
> and data quality question, not an architecture question."*
