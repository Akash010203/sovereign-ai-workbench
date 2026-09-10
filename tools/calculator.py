"""tools/calculator.py — Safe arithmetic calculator tool."""
from __future__ import annotations

import ast
import math
import operator

from tools.registry import BaseTool, ToolResult

# Whitelist of safe operations and functions
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_SAFE_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "abs": abs, "round": round, "pi": math.pi, "e": math.e,
}


def _safe_eval(node):
    """Recursively evaluate an AST node using only whitelisted operations."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsafe constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsafe operator: {type(node.op)}")
        return op_func(_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsafe unary operator: {type(node.op)}")
        return op_func(_safe_eval(node.operand))
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsafe function call.")
        func = _SAFE_FUNCS.get(node.func.id)
        if func is None:
            raise ValueError(f"Function '{node.func.id}' not in calculator whitelist.")
        args = [_safe_eval(a) for a in node.args]
        return func(*args)
    elif isinstance(node, ast.Name):
        val = _SAFE_FUNCS.get(node.id)
        if isinstance(val, (int, float)):
            return val
        raise ValueError(f"Unknown name: {node.id}")
    raise ValueError(f"Unsupported AST node: {type(node)}")


class CalculatorTool(BaseTool):
    """
    Safe arithmetic expression evaluator.

    Uses AST-based evaluation with a strict whitelist — NEVER calls
    eval() or exec() on raw strings. Only arithmetic, math functions,
    and numeric constants are allowed.
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate a mathematical expression safely. "
            "Args: expression (str). Supports +,-,*,/,**,%, sqrt, log, sin, cos, tan, abs, round, pi, e."
        )

    def run(self, expression: str = "") -> ToolResult:
        if not expression.strip():
            return ToolResult(tool_name=self.name, success=False, output=None,
                              error="Empty expression.")
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            result = _safe_eval(tree.body)
            return ToolResult(
                tool_name=self.name, success=True,
                output=result,
                metadata={"expression": expression, "result": result}
            )
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, output=None,
                              error=f"Calculation error: {exc}")
