"""
router/router.py — Task router: classify → select model → return routing decision.

DATA FLOW
---------
User request (str)
    │
    ▼  task_classifier.classify_task()
TaskCategory
    │
    ▼  policies.ROUTING_POLICY[category]
preferred model list + tool list
    │
    ▼  try each model in order; pick first available
    │   (registry.get(name).is_available())
    │
    ▼
RoutingDecision(model, category, tools, reason)
    │
    ▼
Agent / Executor uses the selected model + tools
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from models.adapters.base import ModelProvider
from models.registry import ModelRegistry
from router.task_classifier import TaskCategory, classify_task
from router.policies import ROUTING_POLICY

log = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """The result of routing a user request."""
    category:      TaskCategory
    model:         Optional[ModelProvider]   # None if no model available
    model_name:    str
    tools:         list[str]
    reason:        str
    user_input:    str


class TaskRouter:
    """
    Routes user requests to the appropriate local model and tool set.

    HONESTLY LABELLED: The classification step is rule-based keyword
    matching (not neural).  The model selection step is a priority list
    filtered by availability.

    Args:
        registry: The populated ModelRegistry (Phase 8).
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def route(self, user_input: str) -> RoutingDecision:
        """
        Classify the input, select a model, and return a RoutingDecision.

        Args:
            user_input: The raw user message.

        Returns:
            RoutingDecision with the selected model and active tools.
        """
        # ── 1. Classify (rule-based) ───────────────────────────────────
        category = classify_task(user_input)
        policy   = ROUTING_POLICY.get(category, ROUTING_POLICY[TaskCategory.UNKNOWN])

        log.info("Router: input_len=%d  category=%s", len(user_input), category.value)

        # ── 2. Select model (first available in preference list) ───────
        selected_model: Optional[ModelProvider] = None
        selected_name  = "none"
        reason         = "no model available"

        for model_name in policy["preferred"]:
            adapter = self.registry.get(model_name)
            if adapter is not None and adapter.is_available():
                selected_model = adapter
                selected_name  = model_name
                reason = (
                    f"rule-based routing: category='{category.value}' → "
                    f"selected '{model_name}' (first available in policy)"
                )
                break

        if selected_model is None:
            log.warning(
                "No model available for category '%s'. Preferred: %s",
                category.value, policy["preferred"],
            )
            reason = (
                f"rule-based routing: category='{category.value}' → "
                f"no model available from {policy['preferred']}"
            )

        log.info("Router decision: model=%s  tools=%s", selected_name, policy["tools"])

        return RoutingDecision(
            category=category,
            model=selected_model,
            model_name=selected_name,
            tools=list(policy.get("tools", [])),
            reason=reason,
            user_input=user_input,
        )
