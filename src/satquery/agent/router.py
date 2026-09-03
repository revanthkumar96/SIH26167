"""Query routing: natural language plus input configuration to a task.

Deliberately rule-based rather than an LLM call. Task selection is directly scored
("correct task/tool selection"), and a deterministic router is auditable, instant,
free, and cannot hallucinate a task that does not exist. The matched rule is
recorded so the trace can show *why* a route was chosen.

The input configuration constrains the answer before any keyword is read: a
bi-temporal pair can only produce a change task, whatever the wording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from satquery.schema import InputConfig, Task

_GROUNDING_PATTERNS = (
    r"\bhighlight\b",
    r"\blocate\b",
    r"\bwhere is\b",
    r"\bwhere are\b",
    r"\bmark\b",
    r"\bshow me the\b",
    r"\bpoint (?:to|out)\b",
    r"\bbounding box\b",
    r"\bsegment\b",
    r"\bdelineate\b",
)

_CAPTION_PATTERNS = (
    r"\bdescribe\b",
    r"\bcaption\b",
    r"\bsummar(?:ise|ize)\b",
    r"\bwhat (?:do you|can you) see\b",
    r"\boverview\b",
    r"\bwhat is (?:in|shown in) this image\b",
    r"\btell me about\b",
)

_CHANGE_CAPTION_PATTERNS = (
    r"\bdescribe\b",
    r"\bsummar(?:ise|ize)\b",
    r"\bwhat changed\b",
    r"\bwhat has changed\b",
    r"\bwhat is different\b",
)

_QUESTION_PATTERNS = (
    r"\bhow many\b",
    r"\bis there\b",
    r"\bare there\b",
    r"\bdoes\b",
    r"\bwhat type\b",
    r"\bwhich\b",
    r"\bhas\b",
    r"^\s*(?:is|are|do|can)\b",
)


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """The routing outcome, with the reason recorded for the trace."""

    task: Task
    rule: str
    confidence: float


def _matches(query: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        if re.search(pattern, query):
            return pattern
    return None


def route(query: str, input_config: InputConfig) -> RouteDecision:
    """Classify a query into exactly one task.

    Never fails: an unrecognised query falls back to the most general task the
    input configuration supports, and says so in the rule.
    """
    text = (query or "").strip().lower()

    if input_config is InputConfig.CROSSMODAL_PAIR:
        # Only one task consumes an optical-SAR pair, so wording cannot change it.
        return RouteDecision(Task.CROSSMODAL_VQA, "input_config=crossmodal_pair", 1.0)

    if input_config is InputConfig.BITEMPORAL_PAIR:
        if not text:
            return RouteDecision(Task.CHANGE_CAPTION, "bitemporal + empty query", 0.9)
        matched = _matches(text, _CHANGE_CAPTION_PATTERNS)
        # "What changed?" wants a description; "Has built-up increased?" wants an
        # answer. Both are change tasks, so only the output shape differs.
        if matched and not _matches(text, _QUESTION_PATTERNS):
            return RouteDecision(Task.CHANGE_CAPTION, f"bitemporal + {matched}", 0.85)
        return RouteDecision(Task.CHANGE_VQA, "bitemporal + question", 0.9)

    if not text:
        return RouteDecision(Task.CAPTION, "single + empty query", 0.9)

    matched = _matches(text, _GROUNDING_PATTERNS)
    if matched:
        return RouteDecision(Task.GROUNDING, f"single + {matched}", 0.9)

    matched = _matches(text, _CAPTION_PATTERNS)
    if matched and not _matches(text, _QUESTION_PATTERNS):
        return RouteDecision(Task.CAPTION, f"single + {matched}", 0.85)

    matched = _matches(text, _QUESTION_PATTERNS)
    if matched:
        return RouteDecision(Task.VQA, f"single + {matched}", 0.9)

    if text.endswith("?"):
        return RouteDecision(Task.VQA, "single + question mark", 0.75)

    return RouteDecision(Task.VQA, "single + default", 0.6)
