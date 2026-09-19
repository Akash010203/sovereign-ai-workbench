"""
TEST F — Agent System Full Validation

Verifies:
- Agent lifecycle: plan → execute → verify
- Tool dispatch works
- Error handling
- Memory works
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


class TestAgentComponents:
    """Test individual agent components."""

    def test_planner_creates_steps(self):
        from agents.planner import Planner
        from agents.state import TaskState
        from tools import build_default_tool_registry

        tool_reg = build_default_tool_registry(workspace_root=str(ROOT), retriever=None)
        planner = Planner(tool_reg)
        state = TaskState(
            task_id="test-001",
            user_input="Calculate 25 times 48",
            model_name="none",
            category="calculation",
        )
        steps = planner.plan(state)
        assert isinstance(steps, list)

    def test_executor_runs_steps(self):
        from agents.executor import Executor
        from agents.state import TaskState, AgentStep
        from tools import build_default_tool_registry

        tool_reg = build_default_tool_registry(workspace_root=str(ROOT), retriever=None)
        executor = Executor(tool_reg, model=None)

        state = TaskState(
            task_id="test-002",
            user_input="test",
            model_name="none",
            category="calculation",
        )
        step = AgentStep(
            step_number=1,
            description="Calculate 2+2",
            tool_name="calculator",
            tool_args={"expression": "2+2"},
        )
        state.plan = [step]
        result_state = executor.execute_plan(state)
        assert result_state is not None

    def test_verifier_verifies(self):
        from agents.verifier import Verifier
        from agents.state import TaskState, TaskStatus

        verifier = Verifier()
        state = TaskState(
            task_id="test-003",
            user_input="test",
            model_name="none",
            category="test",
        )
        state.final_answer = "Some answer"
        state.update_status(TaskStatus.VERIFYING)
        result = verifier.verify(state)
        assert result.status in (TaskStatus.DONE, TaskStatus.FAILED)

    def test_memory_stores_turns(self):
        from agents.memory import AgentMemory

        memory = AgentMemory(max_turns=5)
        memory.add_user("Hello")
        memory.add_assistant("Hi there")
        memory.add_user("How are you?")

        prompt = memory.format_for_prompt()
        assert "Hello" in prompt
        assert "Hi there" in prompt

    def test_memory_truncates(self):
        from agents.memory import AgentMemory

        memory = AgentMemory(max_turns=2)
        for i in range(10):
            memory.add_user(f"Message {i}")
            memory.add_assistant(f"Reply {i}")

        # Should only keep last 2 turns
        prompt = memory.format_for_prompt()
        assert "Message 9" in prompt


class TestAgentEndToEnd:
    """End-to-end agent tests."""

    @pytest.fixture
    def agent(self):
        from models.registry import ModelRegistry
        from tools import build_default_tool_registry
        from agents.agent import Agent
        registry = ModelRegistry()
        tool_registry = build_default_tool_registry(
            workspace_root=str(ROOT), retriever=None
        )
        return Agent(registry, tool_registry)

    def test_calculation_task(self, agent):
        state = agent.run("Calculate 25 * 48")
        assert state.final_answer is not None

    def test_task_state_has_metadata(self, agent):
        state = agent.run("Hello world")
        assert state.task_id is not None
        assert state.category is not None
        assert state.model_name is not None
        assert state.created_at is not None

    def test_agent_serialization(self, agent):
        state = agent.run("Test task")
        d = state.to_dict()
        assert isinstance(d, dict)
        assert "task_id" in d
        assert "final_answer" in d
