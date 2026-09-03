"""Default tool set.

The registry is the "predefined registry" the problem statement requires the
controller to select from. Deterministic specialists and VLM tools sit side by side
in it and are selected the same way.
"""

from __future__ import annotations

from satquery.agent.registry import ToolRegistry
from satquery.agent.tools.change import ChangeMaskTool
from satquery.agent.tools.indices import OpticalIndicesTool, SarIndicesTool
from satquery.agent.tools.vlm import VLMTool, build_vlm_tools

__all__ = [
    "ChangeMaskTool",
    "OpticalIndicesTool",
    "SarIndicesTool",
    "VLMTool",
    "build_vlm_tools",
    "default_registry",
]


def default_registry() -> ToolRegistry:
    """Every tool the running system exposes."""
    registry = ToolRegistry()
    registry.register(ChangeMaskTool())
    registry.register(OpticalIndicesTool())
    registry.register(SarIndicesTool())
    for tool in build_vlm_tools():
        registry.register(tool)
    return registry
