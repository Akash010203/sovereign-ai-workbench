"""Regression coverage for agent failure semantics and sandbox policy."""
from agents.executor import Executor
from agents.state import AgentStep, TaskState
from tools import build_default_tool_registry
from tools.code_sandbox import CodeSandboxTool


def test_executor_stops_after_a_failed_step(tmp_path):
    registry = build_default_tool_registry(workspace_root=str(tmp_path))
    state = TaskState(task_id="halt", user_input="test")
    state.plan = [
        AgentStep(1, "bad calculation", "calculator", {"expression": "bad"}),
        AgentStep(2, "must not run", "calculator", {"expression": "2 + 2"}),
    ]

    Executor(registry).execute_plan(state)

    assert state.error
    assert state.plan[0].success is False
    assert state.plan[1].success is None


def test_sandbox_blocks_os_and_file_access_by_default():
    sandbox = CodeSandboxTool()

    assert not sandbox.run("import os").success
    assert not sandbox.run("print(open('secret.txt').read())").success


def test_agent_extracts_a_plain_calculation_expression():
    from agents.agent import Agent

    assert Agent._calculation_expression("Calculate 25 times 48") == "25 * 48"
