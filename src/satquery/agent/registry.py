"""Tool base class and registry.

A tool is a declarative entry, not an ad-hoc call. The registry is what the
controller selects from, and ``ToolSpec.allowed_params`` is enforced on every
invocation: the problem statement permits the agent to configure only sanctioned
parameters, so an out-of-range value is rejected rather than trusted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from satquery.agent.context import RunContext
from satquery.schema import InputConfig, Task, ToolSpec


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool returns. ``outputs`` is copied verbatim into the trace step."""

    outputs: Mapping[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    evidence: tuple[Mapping[str, Any], ...] = ()


class ParameterError(ValueError):
    """Raised when the controller asks a tool for a parameter it does not permit."""


class Tool(ABC):
    """A registry-visible capability with a fixed contract."""

    spec: ToolSpec

    @abstractmethod
    def run(self, ctx: RunContext, **params: Any) -> ToolResult:
        """Execute against the run context. Params are already validated."""

    def invoke(self, ctx: RunContext, **params: Any) -> ToolResult:
        """Validate parameters, then run. The controller always calls this."""
        errors = self.spec.validate_params(params)
        if errors:
            raise ParameterError("; ".join(errors))
        return self.run(ctx, **params)

    @property
    def name(self) -> str:
        return self.spec.name

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "version": self.spec.version,
            "accepts": str(self.spec.accepts),
            "tasks": [str(t) for t in self.spec.tasks],
            "allowed_params": {
                key: sorted(value)
                if isinstance(value, (set, frozenset))
                else list(value)
                for key, value in self.spec.allowed_params.items()
            },
            "outputs": list(self.spec.outputs),
        }


class ToolRegistry:
    """Name-addressed collection of tools, queryable by task and input config."""

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise KeyError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            known = ", ".join(sorted(self._tools)) or "<none>"
            raise KeyError(f"unknown tool '{name}'. Registered: {known}")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def for_task(self, task: Task, input_config: InputConfig) -> list[Tool]:
        """Every tool that can serve this task for this input configuration."""
        return [
            tool
            for tool in self._tools.values()
            if task in tool.spec.tasks and tool.spec.accepts is input_config
        ]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict[str, Any]]:
        return [self._tools[name].describe() for name in self.names()]

    def __len__(self) -> int:
        return len(self._tools)
