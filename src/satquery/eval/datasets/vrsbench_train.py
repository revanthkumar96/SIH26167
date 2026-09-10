"""VRSBench **train** split -- a different shape from the EVAL files.

The EVAL release ships three tidy per-task JSONs with ``image_id`` / ``question``
/ ``ground_truth`` keys. ``VRSBench_train.json`` ships nothing of the kind: it is
a single 65 MB LLaVA-style instruction file, 142,390 two-turn conversations over
20,262 images, with all three tasks interleaved and distinguished only by a
``[caption]`` / ``[refer]`` / ``[vqa]`` tag inside the human turn. Its ``id``
field is the constant string ``Final_Data/v1.2`` on every record, so it is not an
identifier and cannot be used as one.

Pointing the EVAL adapters at it does not fail loudly -- it fails on a missing
``image_id`` key, which reads like a release-version problem and invites someone
to "fix" it with a field override that then produces nonsense. Hence a separate
adapter.

Three things this has to undo, none of them cosmetic:

*The instruction wrappers.* The published prompts are GeoChat-style paraphrases
("Use the provided image to answer the question: ... Provide your answer as short
as possible."). We do not train on those, because the model must meet
``eval/prompts.py``'s phrasing at inference. Only the question inside the wrapper
is real; the wrapper is discarded and the prompt re-rendered by ``build_prompt``.

*The referring wrapper.* The expression is always inside ``<p>...</p>`` -- all
36,313 of them -- with conversational scaffolding around it that varies.

*The box grid.* Answers are ``{<x1><y1><x2><y2>}`` on a 0-100 grid, while our
grounding prompt asks for 0-1000. About 8.8% of coordinates fall outside 0-100
(min -73, max 196) because the published boxes run past the image edge, so
conversion clamps. Reading these as anything other than 0-100 silently produces
boxes in the wrong place, and grounding is already the weakest column -- a units
bug there would be invisible among the genuine failures.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, ClassVar

from satquery.eval.datasets.base import BenchmarkDataset, as_records, register
from satquery.schema import ImageRef, Modality, Sample, Task

#: Task tag at the head of every human turn, after the ``<image>`` token.
_TAG = re.compile(r"\[(caption|refer|vqa)\]\s*", re.I)

#: The image placeholder the published file carries. Our prompts do not use one.
_IMAGE_TOKEN = re.compile(r"<image>\s*")

#: Referring expressions are always wrapped, without exception in v1.2.
_EXPRESSION = re.compile(r"<p>(.*?)</p>", re.S)

#: ``{<45><45><59><59>}`` -- a 0-100 grid, negatives and >100 both occurring.
_BOX = re.compile(r"\{\s*<(-?\d+)>\s*<(-?\d+)>\s*<(-?\d+)>\s*<(-?\d+)>\s*\}")

#: The published grid. Not 1000: see the module docstring.
BOX_GRID = 100.0

#: Instruction wrappers, stripped from each end independently rather than matched
#: as whole templates. The published file mixes them freely -- ``Question: {q}``
#: appears both with the ``Short answer:`` suffix and without it, 8,443 times --
#: so a set of fixed circumfix templates leaves the unpaired half in the question
#: and trains the model on "Question: is there a vehicle?". Enumerated from the
#: file itself, not guessed.
_VQA_PREFIXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^The question\s+", re.I),
    re.compile(
        r"^Based on the image,?\s*respond to this question with a short answer\s*:\s*",
        re.I,
    ),
    re.compile(r"^Use the provided image to answer the question\s*:\s*", re.I),
    re.compile(
        r"^Given the image,?\s*answer the following question with no more than\s+"
        r"[\w-]+\s+words\.?\s*",
        re.I,
    ),
    re.compile(r"^(?:Question|Q)\s*:\s*", re.I),
    re.compile(r"^Answer the question(?:\s+below)?\s*:\s*", re.I),
)

#: A full stop left stranded after the question's own terminator.
_DANGLING_STOP = re.compile(r"(?<=[?!])\s*\.+$")

_VQA_SUFFIXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\s*can be answered using the image\.?\s*A short answer is\.?$", re.I),
    re.compile(r"\s*Provide your answer as short as possible\.?$", re.I),
    re.compile(r"\.?\s*A short answer to the question is\.?$", re.I),
    re.compile(r"\.?\s*A short answer is\.?$", re.I),
    re.compile(r"\s*Short answer\s*:\s*\.?$", re.I),
    re.compile(r"\s*A\s*:\s*\.?$"),
)


class VRSBenchTrainError(ValueError):
    """The train file is not in the shape this adapter understands."""


def strip_question(text: str) -> str:
    """Recover the bare question from a published instruction wrapper.

    Each end is stripped at most once. Text with no wrapper is returned as it
    stands, which is correct -- a large share of records carry the question
    unadorned -- and a strip that would empty the string is refused, so a
    question that happens to read like scaffolding survives rather than becoming
    an empty prompt.
    """
    cleaned = text.strip()
    for patterns in (_VQA_PREFIXES, _VQA_SUFFIXES):
        for pattern in patterns:
            stripped = pattern.sub("", cleaned, count=1).strip()
            if stripped != cleaned and stripped:
                cleaned = stripped
                break
    # Several templates append their own full stop after the question mark
    # ("...a short answer: What is the main object?."). Left in place it is a
    # difference the model sees on every record from that template.
    return _DANGLING_STOP.sub("", cleaned)


def parse_box(text: str) -> tuple[float, float, float, float] | None:
    """``{<x1><y1><x2><y2>}`` on the 0-100 grid to unit floats.

    Clamped, because 8.8% of published coordinates fall outside the grid where a
    box runs past the image edge. Returns ``None`` for a box that is degenerate
    once clamped -- an empty region is not a grounding target, and training on
    one teaches the model to emit a point when asked to localise.
    """
    match = _BOX.search(text or "")
    if not match:
        return None
    x1, y1, x2, y2 = (int(g) / BOX_GRID for g in match.groups())
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    box = tuple(min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box  # type: ignore[return-value]


@register("vrsbench_train")
class VRSBenchTrain(BenchmarkDataset):
    """The interleaved train file, filtered to one task at a time.

    ``extra.task_tag`` selects which of the three the config wants. One tag per
    config, matching how the EVAL side is organised, so a train corpus and its
    benchmark are configured the same way and capped independently.
    """

    default_fields: ClassVar[dict[str, tuple[str, ...]]] = {
        "image_id": ("image", "image_id", "img_id", "filename"),
    }

    #: Which schema ``Task`` each published tag becomes.
    TAGS: ClassVar[dict[str, Task]] = {
        "caption": Task.CAPTION,
        "refer": Task.GROUNDING,
        "vqa": Task.VQA,
    }

    def _tag(self) -> str:
        tag = str(self.config.extra.get("task_tag", "")).strip().lower()
        if tag not in self.TAGS:
            known = ", ".join(sorted(self.TAGS))
            raise VRSBenchTrainError(
                f"{self.config.name}: set extra.task_tag to one of {known}; "
                f"got {tag!r}. The train file interleaves all three tasks in one "
                f"file, so an adapter that did not filter would mix caption "
                f"targets into a VQA config."
            )
        return tag

    def _turns(self, record: dict[str, Any], index: int) -> tuple[str, str]:
        turns = record.get("conversations") or []
        if len(turns) != 2:
            raise VRSBenchTrainError(
                f"{self.config.name}: record {index} has {len(turns)} turn(s); "
                f"the train file is two-turn throughout"
            )
        return str(turns[0].get("value", "")), str(turns[1].get("value", ""))

    def _tagged(self) -> Iterator[tuple[int, dict[str, Any], str, str, str]]:
        """Yield ``(index, record, tag, human, gpt)`` for every parsable row."""
        for index, record in enumerate(as_records(self.read_json())):
            human, gpt = self._turns(record, index)
            match = _TAG.search(human)
            if not match:
                continue  # untagged rows do not occur in v1.2; skip rather than guess
            body = _IMAGE_TOKEN.sub("", human[match.end() :]).strip()
            yield index, record, match.group(1).lower(), body, gpt.strip()

    def load(self) -> list[Sample]:
        tag = self._tag()
        task = self.TAGS[tag]
        samples: list[Sample] = []

        for index, record, found, body, answer in self._tagged():
            if found != tag:
                continue
            image = ImageRef(
                path=self.resolve_image(self.require(record, "image_id")),
                modality=Modality.RGB,
            )
            # Ids are index-based because the file's own `id` field is a single
            # constant across all 142,390 records.
            sample_id = f"{self.config.name}-{index}"

            if task is Task.GROUNDING:
                found_expression = _EXPRESSION.search(body)
                box = parse_box(answer)
                # Six records in v1.2 ship an empty `<p></p>` -- a box with
                # nothing to describe. Dropped rather than passed on: a
                # grounding prompt with no description asks the model to locate
                # the empty string, and build_prompt rejects it anyway.
                expression = (
                    found_expression.group(1).strip() if found_expression else ""
                )
                if not expression or box is None:
                    continue
                samples.append(
                    Sample(
                        sample_id=sample_id,
                        task=task,
                        images=(image,),
                        question=expression,
                        bbox=box,
                    )
                )
            elif task is Task.CAPTION:
                if not answer:
                    continue
                # The published instruction is a paraphrase and is discarded --
                # build_prompt renders the caption prompt the harness uses.
                samples.append(
                    Sample(
                        sample_id=sample_id,
                        task=task,
                        images=(image,),
                        answer=answer,
                        references=(answer,),
                    )
                )
            else:
                question = strip_question(body)
                if not question or not answer:
                    continue
                samples.append(
                    Sample(
                        sample_id=sample_id,
                        task=task,
                        images=(image,),
                        question=question,
                        answer=answer,
                    )
                )
        return samples
