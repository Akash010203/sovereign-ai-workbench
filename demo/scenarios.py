"""
demo/scenarios.py — SIH 2026 Problem Statement SIH 117 demo scenarios.

WHAT THIS FILE IS
-----------------
This file defines the 5 key demonstration scenarios that will be run
live in front of the SIH 2026 judges.  Each scenario:
  1. Is a realistic industrial use case (oil & gas / plant operations)
  2. Exercises a different set of platform capabilities
  3. Runs completely offline on the demo machine
  4. Is self-contained and can be run with a single function call

THE 5 DEMO SCENARIOS
--------------------
1. OCR + Document Analysis
   "Read this scanned maintenance logbook page and identify action items."
   → Tesseract OCR → custom MiniLLM analysis → Word report generated

2. RAG + Knowledge Retrieval
   "What does our SOP say about replacing a faulty compressor seal?"
   → Local vector search → cited answer → no hallucination

3. Code Generation + Sandboxed Execution
   "Write a Python script to plot the pressure sensor data from this CSV."
   → custom MiniLLM → sandbox execution → output captured

4. Engineering Calculation
   "Calculate the pressure drop across a 50m pipeline, d=0.1m, flow=0.05 m³/s, μ=0.001 Pa·s."
   → AST-safe calculator → Darcy-Weisbach formula → numeric answer

5. Air-Gap Proof
   "Show that no data is leaving this machine."
   → NetworkMonitor → offline verification → timestamped proof log

HOW TO RUN DURING THE DEMO
---------------------------
    python demo\\run_demo.py --scenario 1
    python demo\\run_demo.py --scenario 2
    python demo\\run_demo.py --all
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DemoScenario:
    """Definition of one SIH demo scenario."""
    number:      int
    title:       str
    description: str
    user_input:  str
    tools_used:  list[str]
    models_used: list[str]
    what_it_proves: list[str]
    expected_output_keys: list[str] = field(default_factory=list)


DEMO_SCENARIOS = [

    DemoScenario(
        number=1,
        title="OCR + Automated Document Analysis",
        description=(
            "Simulate: A plant engineer scans a handwritten maintenance log. "
            "The system OCRs it locally, identifies action items, and auto-generates "
            "a formal Word report — without any manual copy-paste or cloud API."
        ),
        user_input=(
            "The following text was extracted from a scanned maintenance logbook:\n\n"
            "Date: 10-Sep-2026. Pump P-201 showing vibration at bearing end. "
            "Recommend immediate lubrication and schedule full inspection by 15-Sep. "
            "Oil level low in reservoir tank T-04. Replenish with ISO 46 hydraulic oil. "
            "Operator signature: R. Sharma.\n\n"
            "Identify all action items and generate a formal approval note."
        ),
        tools_used=["ocr", "word_write"],
        models_used=["custom_minilm_v1"],
        what_it_proves=[
            "Local OCR pipeline works without cloud services",
            "From-scratch MiniLLM processes domain-specific text in-process",
            "Word document is generated and saved to disk",
            "Full pipeline: scan → extract → analyze → output",
        ],
    ),

    DemoScenario(
        number=2,
        title="RAG: Knowledge Base Retrieval with Citations",
        description=(
            "Simulate: An engineer asks a question answered by a procedure manual "
            "that has been ingested into the local knowledge base. "
            "The system retrieves the relevant chunk and cites the source."
        ),
        user_input="What is the correct procedure for replacing a faulty compressor seal?",
        tools_used=["rag_search"],
        models_used=["custom_minilm_v1"],
        what_it_proves=[
            "Local vector index (no Pinecone, no Chroma, no cloud) returns relevant results",
            "Answer includes source citations (prevents hallucination)",
            "Zero external API calls during retrieval",
            "RAG chunking, embedding, and retrieval all custom-built",
        ],
    ),

    DemoScenario(
        number=3,
        title="Code Generation + Sandboxed Execution",
        description=(
            "Simulate: A data engineer asks the system to write and run a "
            "Python data analysis script. The code runs in an isolated sandbox — "
            "not the main process — proving safety."
        ),
        user_input=(
            "Write a Python script that calculates the average, max, and min "
            "of this list of pressure readings: [45.2, 47.1, 43.8, 46.5, 48.0, 44.9, 45.7]. "
            "Then run it and show the output."
        ),
        tools_used=["code_sandbox"],
        models_used=["custom_minilm_v1"],
        what_it_proves=[
            "LLM generates correct domain-specific Python code",
            "Code executes in an isolated subprocess (not the main process)",
            "stdout/stderr are captured and returned as structured data",
            "Sandbox has a timeout — infinite loops are killed automatically",
        ],
    ),

    DemoScenario(
        number=4,
        title="Engineering Calculation (Safe AST Calculator)",
        description=(
            "Simulate: An engineer needs to calculate pressure drop using "
            "the Darcy-Weisbach equation. The system evaluates the formula "
            "safely without eval() or exec()."
        ),
        user_input=(
            "Calculate the pressure drop using Darcy-Weisbach: "
            "ΔP = (f × L × ρ × v²) / (2 × D). "
            "Given: f=0.02 (Darcy friction factor), L=50m (pipe length), "
            "ρ=1000 kg/m³ (water), v=0.637 m/s (velocity), D=0.1m (diameter). "
            "So: (0.02 * 50 * 1000 * 0.637 * 0.637) / (2 * 0.1)"
        ),
        tools_used=["calculator"],
        models_used=["custom_minilm_v1"],
        what_it_proves=[
            "Safe AST-based calculator — no eval() or exec() vulnerabilities",
            "Handles real engineering formulas correctly",
            "Result is deterministic and verifiable",
        ],
    ),

    DemoScenario(
        number=5,
        title="Air-Gap Proof: Zero External Connections",
        description=(
            "Demonstrate to the judges that no data is leaving the machine "
            "at runtime. The network monitor intercepts all Python-level TCP "
            "connections and reports external connection attempts."
        ),
        user_input="Run an air-gap verification and show the network activity report.",
        tools_used=[],
        models_used=[],
        what_it_proves=[
            "No external TCP connections attempted during operation",
            "The custom MiniLLM runs in-process — no API calls",
            "Timestamped proof log saved to disk",
        ],
    ),
]


def get_scenario(number: int) -> DemoScenario:
    for s in DEMO_SCENARIOS:
        if s.number == number:
            return s
    raise ValueError(f"Scenario {number} not found. Valid: 1-{len(DEMO_SCENARIOS)}")
