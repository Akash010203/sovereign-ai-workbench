# SovereignAI Test Report — Cycle Summary

**Timestamp:** 2026-09-18T18:44:31.138796
**Duration:** 400.8s

## Scoreboard

| Category | Status | Passed | Failed |
|---|---|---|---|
| Environment / Startup | PASS | 94 | 0 |
| Tokenizer | PASS | 36 | 0 |
| Custom LLM | PASS | 30 | 0 |
| Training Pipeline | PASS | 5 | 0 |
| Router | FAIL | 23 | 1 |
| Agent System | FAIL | 7 | 1 |
| RAG | FAIL | 12 | 2 |
| OCR | PASS | 4 | 0 |
| Document Generation | PASS | 11 | 0 |
| Code Sandbox | PASS | 14 | 0 |
| Database | PASS | 11 | 0 |
| Security | FAIL | 15 | 3 |
| Air-Gap / Offline | FAIL | 2 | 1 |
| Backend API | BLOCKED | 0 | 0 |
| Performance | FAIL | 6 | 1 |
| Fake Implementation Detection | FAIL | 8 | 1 |

## Totals

- **Total tests:** 288
- **Passed:** 278
- **Failed:** 10
- **Errors:** 0
- **Blocked:** 1

## Failures

### Router

```
zone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    self.updated_at = datetime.utcnow().isoformat()

tests/system/test_router_full.py::TestAgentSystem::test_agent_run_returns_task_state
tests/system/test_router_full.py::TestAgentSystem::test_agent_run_returns_task_state
tests/system/test_router_full.py::TestAgentSystem::test_agent_handles_empty_input
tests/system/test_router_full.py::TestAgentSystem::test_agent_produces_final_answer
tests/system/test_router_full.py::TestAgentSystem::test_agent_produces_final_answer
tests/system/test_router_full.py::TestAgentSystem::test_agent_planner_creates_steps
tests/system/test_router_full.py::TestAgentSystem::test_agent_planner_creates_steps
  C:\Users\akash\Videos\Captures\sovereign-ai-workbench\agents\state.py:42: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    timestamp:    str = field(default_factory=lambda: datetime.utcnow().isoformat())

tests/system/test_router_full.py::TestAgentSystem::test_agent_run_returns_task_state
tests/system/test_router_full.py::TestAgentSystem::test_agent_produces_final_answer
tests/system/test_router_full.py::TestAgentSystem::test_agent_planner_creates_steps
  C:\Users\akash\Videos\Captures\sovereign-ai-workbench\tools\registry.py:30: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    timestamp:  str = field(default_factory=lambda: datetime.utcnow().isoformat())

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/system/test_router_full.py::TestRouterIntegration::test_router_no_model_available
1 failed, 23 passed, 34 warnings in 2.75s

```

### Agent System

```
    str = field(default_factory=lambda: datetime.utcnow().isoformat())

tests/system/test_agent_full.py::TestAgentComponents::test_planner_creates_steps
tests/system/test_agent_full.py::TestAgentComponents::test_planner_creates_steps
tests/system/test_agent_full.py::TestAgentEndToEnd::test_calculation_task
tests/system/test_agent_full.py::TestAgentEndToEnd::test_calculation_task
tests/system/test_agent_full.py::TestAgentEndToEnd::test_task_state_has_metadata
tests/system/test_agent_full.py::TestAgentEndToEnd::test_agent_serialization
  C:\Users\akash\Videos\Captures\sovereign-ai-workbench\agents\state.py:42: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    timestamp:    str = field(default_factory=lambda: datetime.utcnow().isoformat())

tests/system/test_agent_full.py: 14 warnings
  C:\Users\akash\Videos\Captures\sovereign-ai-workbench\agents\state.py:64: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    self.updated_at = datetime.utcnow().isoformat()

tests/system/test_agent_full.py::TestAgentEndToEnd::test_calculation_task
  C:\Users\akash\Videos\Captures\sovereign-ai-workbench\tools\registry.py:30: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    timestamp:  str = field(default_factory=lambda: datetime.utcnow().isoformat())

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/system/test_agent_full.py::TestAgentComponents::test_executor_runs_steps
1 failed, 7 passed, 33 warnings in 2.65s

```

### RAG

```
 any("pump" in r["text"].lower() or "P-101" in r["text"] for r in results)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\system\test_rag_full.py:188: in <genexpr>
    assert any("pump" in r["text"].lower() or "P-101" in r["text"] for r in results)
                         ^^^^^^^^^
E   TypeError: 'RetrievalResult' object is not subscriptable
---------------------------- Captured stderr call -----------------------------
\rLoading weights:   0%|          | 0/103 [00:00<?, ?it/s]\rLoading weights: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 103/103 [00:00<00:00, 6959.31it/s]\n\rLoading weights:   0%|          | 0/103 [00:00<?, ?it/s]\rLoading weights: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 103/103 [00:00<00:00, 6146.59it/s]
____________________ TestRAGCitations.test_citation_format ____________________
tests\system\test_rag_full.py:198: in test_citation_format
    formatted = format_citations(results)
                ^^^^^^^^^^^^^^^^^^^^^^^^^
rag\citations.py:13: in format_citations
    lines.append(f"  [{i}] {r.source} (relevance: {r.score:.2f})")
                            ^^^^^^^^
E   AttributeError: 'dict' object has no attribute 'source'
============================== warnings summary ===============================
tests/system/test_rag_full.py: 14 warnings
  C:\Users\akash\AppData\Local\Programs\Python\Python314\Lib\site-packages\torch\jit\_script.py:359: DeprecationWarning: `torch.jit.script_method` is not supported in Python 3.14+ and may break. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/system/test_rag_full.py::TestRAGRetriever::test_retriever_end_to_end
FAILED tests/system/test_rag_full.py::TestRAGCitations::test_citation_format
2 failed, 12 passed, 14 warnings in 24.58s

```

### Security

```
l.py::TestNetworkMonitor::test_check_internet_returns_dict
tests/system/test_security_full.py::TestOfflineMode::test_verify_offline_returns_dict
tests/system/test_security_full.py::TestOfflineMode::test_verify_offline_has_required_keys
  C:\Users\akash\Videos\Captures\sovereign-ai-workbench\security\network_monitor.py:85: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    "timestamp":     datetime.utcnow().isoformat(),

tests/system/test_security_full.py::TestAuditLogging::test_log_event
  C:\Users\akash\Videos\Captures\sovereign-ai-workbench\security\audit.py:36: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    "timestamp":  datetime.utcnow().isoformat(),

tests/system/test_security_full.py::TestOfflineMode::test_verify_offline_returns_dict
tests/system/test_security_full.py::TestOfflineMode::test_verify_offline_has_required_keys
  C:\Users\akash\Videos\Captures\sovereign-ai-workbench\security\offline_mode.py:26: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    "timestamp":         datetime.utcnow().isoformat(),

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/system/test_security_full.py::TestPathTraversal::test_filesystem_tool_blocks_traversal
FAILED tests/system/test_security_full.py::TestPathTraversal::test_filesystem_tool_blocks_absolute_escape
FAILED tests/system/test_security_full.py::TestSourceCodeNetworkScan::test_no_telemetry_in_source
3 failed, 15 passed, 13 warnings in 4.09s

```

### Air-Gap / Offline

```
st/a", "score": 0.9},
E     tests\test_fineweb_knowledge.py:16 [URL reference] {"text": "beta " * 30, "url": "https://example.test/b", "score": 0.8},
E     tools\ocr.py:6 [URL reference] Install: https://github.com/UB-Mannheim/tesseract/wiki (Windows installer)
E     tests\system\test_airgap.py:27 [requests import] (r'import requests\b', "requests import"),
E     tests\system\test_airgap.py:28 [requests import] (r'from requests ', "requests import"),
E     tests\system\test_airgap.py:29 [urllib import] (r'import urllib', "urllib import"),
E     tests\system\test_airgap.py:30 [aiohttp import] (r'import aiohttp', "aiohttp import"),
E     tests\system\test_airgap.py:31 [httpx import] (r'import httpx', "httpx import"),
E     tests\system\test_sandbox_full.py:114 [requests import] result = tool.run(code="import requests; print('FAIL')")
E     models\adapters\openweight_adapter.py:26 [URL reference] 1. Install Ollama: https://ollama.ai (Windows MSI)
E     models\adapters\openweight_adapter.py:28 [URL reference] 3. Ollama runs as a local server on http://localhost:11434
E     models\adapters\openweight_adapter.py:51 [URL reference] This uses Ollama's local REST API (http://localhost:11434).
E     models\adapters\openweight_adapter.py:60 [URL reference] OLLAMA_DEFAULT_URL = "http://localhost:11434"
E     models\adapters\openweight_adapter.py:86 [urllib import] import urllib.request, json as _json
E     models\adapters\openweight_adapter.py:105 [urllib import] import urllib.request
E     models\custom_minilm\ffn.py:55 [URL reference] https://arxiv.org/abs/2002.05202
E     app\backend\app.py:6 [URL reference] Runs on http://localhost:5000 — NOT exposed to the internet.
E     app\backend\app.py:377 [URL reference] log.info("Starting SovereignAI backend on http://localhost:%d", port)
=========================== short test summary info ===========================
FAILED tests/system/test_airgap.py::TestAirGapSourceScan::test_classify_all_network_deps
1 failed, 2 passed in 0.31s

```

### Performance

```
..F....                                                                  [100%]
================================== FAILURES ===================================
___________________ TestGPUDetection.test_vram_if_available ___________________
tests\system\test_performance.py:41: in test_vram_if_available
    total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'. Did you mean: 'total_memory'?
=========================== short test summary info ===========================
FAILED tests/system/test_performance.py::TestGPUDetection::test_vram_if_available
1 failed, 6 passed in 6.31s

```

### Fake Implementation Detection

```
......F..                                                                [100%]
================================== FAILURES ===================================
_________________ TestFakeDetection.test_no_fake_model_names __________________
tests\system\test_fake_detection.py:133: in test_no_fake_model_names
    assert len(findings) == 0, f"Fake model references:\n" + "\n".join(findings)
E   AssertionError: Fake model references:
E     evaluation\metrics.py: references 'gpt-4'
E   assert 1 == 0
E    +  where 1 = len(["evaluation\\metrics.py: references 'gpt-4'"])
=========================== short test summary info ===========================
FAILED tests/system/test_fake_detection.py::TestFakeDetection::test_no_fake_model_names
1 failed, 8 passed in 8.45s

```
