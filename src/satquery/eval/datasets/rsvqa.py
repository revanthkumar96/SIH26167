"""RSVQA adapter (LR and HR).

RSVQA splits questions, answers and images across separate JSON files joined on id,
rather than shipping one flat annotation list. Records carry an ``active`` flag; the
official protocol evaluates only active entries, so inactive ones are dropped.

Config example (see ``configs/bench/rsvqa_lr.yaml``)::

    extra:
      questions: LR/LR_split_test_questions.json
      answers:   LR/LR_split_test_answers.json
"""

from __future__ import annotations

from typing import Any, ClassVar

from satquery.eval.datasets.base import BenchmarkDataset, as_records, register
from satquery.schema import ImageRef, Modality, Sample, Task


@register("rsvqa")
class RSVQA(BenchmarkDataset):
    default_fields: ClassVar[dict[str, tuple[str, ...]]] = {
        "question": ("question",),
        "qtype": ("type", "question_type"),
        "image_id": ("img_id", "image_id", "image"),
        "question_id": ("id",),
        "answer": ("answer",),
        "answer_question_id": ("question_id",),
    }

    def _path(self, key: str) -> Any:
        rel = self.config.extra.get(key)
        if not rel:
            raise KeyError(
                f"{self.config.name}: extra.{key} is required for the rsvqa adapter"
            )
        return self.config.root / rel

    def _answers_by_question(self) -> dict[Any, str]:
        payload = self.read_json(self._path("answers"))
        mapping: dict[Any, str] = {}
        for record in as_records(payload, ("answers",)):
            if record.get("active") is False:
                continue
            question_id = self.pick(record, "answer_question_id")
            answer = self.pick(record, "answer")
            if question_id is not None and answer is not None:
                mapping.setdefault(question_id, str(answer))
        return mapping

    def load(self) -> list[Sample]:
        answers = self._answers_by_question()
        payload = self.read_json(self._path("questions"))

        samples: list[Sample] = []
        for record in as_records(payload, ("questions",)):
            if record.get("active") is False:
                continue
            question_id = self.pick(record, "question_id")
            answer = answers.get(question_id)
            if answer is None:
                continue
            image_id = self.require(record, "image_id")
            samples.append(
                Sample(
                    sample_id=f"{self.config.name}-{question_id}",
                    task=Task.VQA,
                    images=(
                        ImageRef(
                            path=self.resolve_image(image_id),
                            modality=Modality.OPTICAL,
                        ),
                    ),
                    question=str(self.require(record, "question")),
                    answer=answer,
                    qtype=(
                        str(self.pick(record, "qtype"))
                        if self.pick(record, "qtype")
                        else None
                    ),
                    meta={"image_id": image_id},
                )
            )
        return samples
