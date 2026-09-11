# Demonstration Scenarios & Evaluation Runbook

This runbook outlines the automated demonstration scenarios built into the **Sovereign AI Workbench** to demonstrate full sovereignty, tool integration, RAG, and agentic workflows.

---

## 1. Quick Demonstration Commands

```powershell
# List all available scenarios
python scripts/run_demo.py --list

# Run individual scenarios
python scripts/run_demo.py --scenario 1   # OCR + Document Generation
python scripts/run_demo.py --scenario 2   # RAG Retrieval with Citations
python scripts/run_demo.py --scenario 3   # Python Sandbox Code Execution
python scripts/run_demo.py --scenario 4   # Safe AST Engineering Calculation
python scripts/run_demo.py --scenario 5   # Air-Gap & Zero Network Proof

# Run all scenarios sequentially
python scripts/run_demo.py --all
```

---

## 2. Detailed Scenario Breakdown

### Scenario 1: OCR & Document Extraction
- **Objective**: Ingest a scanned maintenance log or technical diagram.
- **Workflow**: `tools/ocr.py` extracts raw text from the image locally via Tesseract $\rightarrow$ LLM summarizes defects $\rightarrow$ `tools/word.py` outputs a structured DOCX report.

### Scenario 2: RAG Knowledge Retrieval with Grounded Citations
- **Objective**: Answer operational engineering questions using local documents without external API calls.
- **Workflow**: User queries maintenance intervals $\rightarrow$ `rag/retriever.py` queries local vector index $\rightarrow$ returns Top-K chunks with exact paragraph and document citations.

### Scenario 3: Sandboxed Code Execution
- **Objective**: Perform complex data transformations using a Python script.
- **Workflow**: Agent generates Python script $\rightarrow$ `tools/code_sandbox.py` executes script in an isolated subprocess with timeout and memory limits $\rightarrow$ returns structured results.

### Scenario 4: Safe AST Engineering Calculation
- **Objective**: Execute complex mathematical formulas (e.g., Darcy-Weisbach friction head loss) without LLM arithmetic hallucinations.
- **Workflow**: Agent extracts formula and inputs $\rightarrow$ `tools/calculator.py` parses AST safely without `eval()` $\rightarrow$ `agents/verifier.py` checks bounds $\rightarrow$ returns verified answer.

### Scenario 5: Air-Gap Proof & Network Monitor Audit
- **Objective**: Prove zero outbound internet calls occurred.
- **Workflow**: `security/network_monitor.py` asserts zero external network egress during all demo operations and displays timestamped evidence.
