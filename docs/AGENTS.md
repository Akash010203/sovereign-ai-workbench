# Sovereign Agent Architecture & Orchestration

The **Sovereign AI Workbench** includes an autonomous agentic framework designed to plan, execute, and verify complex multi-step industrial and engineering workflows locally.

---

## 1. Architectural Philosophy: Plan → Execute → Verify

Standard LLM completions are one-shot predictions prone to cascading reasoning errors. Sovereign AI workbench decouples reasoning into three distinct, observable components:

```
User Task
    │
    ▼
┌──────────────┐
│   Planner    │  Decomposes task into structured, sequential steps
└───────┬──────┘
        │ Execution Plan
        ▼
┌──────────────┐
│   Executor   │  Calls registered tools, queries RAG, queries SQL, generates responses
└───────┬──────┘
        │ Execution Results
        ▼
┌──────────────┐
│   Verifier   │  Checks numerical correctness, constraint satisfaction, source citations
└───────┬──────┘
        ├── [PASS] ──► Final Validated Answer
        └── [FAIL] ──► Replanning / Self-Correction Loop
```

---

## 2. Directory Structure

The agent module is located in `agents/`:

```
agents/
├── planner.py    # Structured plan generation (steps, tools needed, prerequisites)
├── executor.py   # Multi-step tool executor with error catching
├── verifier.py   # Output verification, AST calculation check, hallucination safeguard
├── memory.py     # Short-term and episodic agent scratchpad
├── state.py      # Serializable execution state tracking
└── agent.py      # Unified agent interface
```

---

## 3. Component Details

### A. Planner (`agents/planner.py`)
Analyzes the user's intent and generates a formal execution graph:
```python
@dataclass
class PlanStep:
    step_id: int
    description: str
    tool_name: Optional[str]
    tool_args: dict
    expected_output: str
```

### B. Executor (`agents/executor.py`)
Iterates through plan steps, binds inputs dynamically, and invokes capabilities via the centralized `tools/registry.py`.

### C. Verifier (`agents/verifier.py`)
Enforces output sanity:
- **Numerical consistency**: Ensures numbers cited in text match tool calculations.
- **Source grounding**: Ensures claims match retrieved RAG chunks.
- **Schema compliance**: Ensures structured outputs adhere to defined JSON/SQL schemas.

### D. Memory & State (`agents/memory.py`, `agents/state.py`)
Maintains execution context across steps, allowing subsequent tool calls to reference results from earlier operations without polluting the language model context window.
