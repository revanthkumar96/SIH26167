#!/usr/bin/env python
"""LoRA adaptation of Qwen3-VL on the prepared corpora.

    # stage A -- domain and cross-modal adaptation
    python scripts/train_lora.py --stage a \
        --data data/prepared/train/train.jsonl \
        --out  runs/adapters/stage-a

    # stage B -- the task mixture, resumed from stage A
    python scripts/train_lora.py --stage b \
        --data data/prepared/mixture/train.jsonl \
        --resume runs/adapters/stage-a --out runs/adapters/stage-b

Plain ``transformers`` + ``peft``; no LLaMA-Factory and no ms-swift. Feasibility
is settled -- a third party trained this exact base on a Tesla T4 (14.56 GiB,
fp16 only), so a 24 GB A10G is roomy, and bf16 removes the GradScaler their path
needed.

Serve the result with vLLM so base and adapted are two model names on one
process, which is the shape the benchmark matrix consumes:

    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-VL-2B-Instruct \
        --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
        --enable-lora --lora-modules qwen3-vl-satquery=runs/adapters/stage-b

Do not run this before the baseline exists. Without a before column, adaptation
is an unverifiable claim.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

try:  # the Trainer invokes the whole callback interface, not just our hook
    from transformers import TrainerCallback as _TrainerCallbackBase
except Exception:  # transformers is only needed on the training host
    _TrainerCallbackBase = object  # type: ignore[assignment,misc]

from satquery.finetune.config import BASE_MODEL, BASE_REVISION, stage_a, stage_b


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} is empty")
    return records


def evidence_share(records: list[dict[str, Any]]) -> float:
    """Fraction of records carrying a measurement preamble.

    Printed at the start of every run and written into the adapter metadata,
    because it is the setting most likely to be wrong in a way nothing else
    would reveal. A corpus prepared without preambles trains a model that has
    never seen the prompt shape it meets at inference, and the only symptom is
    scores that are quietly worse.
    """
    if not records:
        return 0.0
    carried = sum(1 for r in records if str(r.get("evidence_kind", "none")) != "none")
    return carried / len(records)


class ShareGPTDataset:
    """Prepared records, rendered through the processor's own chat template.

    Loss is masked to the assistant turn: training on the prompt tokens teaches
    the model to generate questions, which is not the task and dilutes the
    gradient that matters.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        processor: Any,
        image_root: Path,
        max_length: int,
    ) -> None:
        self.records = records
        self.processor = processor
        self.image_root = image_root
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        record = self.records[index]
        turns = record["conversations"]
        images = [
            Image.open(self.image_root / name).convert("RGB")
            for name in record.get("images", [])
        ]

        content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": turns[0]["value"]})
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        answer = turns[1]["value"]

        batch = self.processor(
            text=[prompt + answer],
            images=images or None,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        # Only the text tensors carry a batch dimension to strip. pixel_values
        # is (patches, dim) and image_grid_thw is (images, 3) -- already per
        # image, not per batch. Indexing [0] on those collapses grid_thw to 1-D
        # and the vision tower fails much later with "too many indices".
        text_keys = {"input_ids", "attention_mask", "token_type_ids"}
        item = {k: (v[0] if k in text_keys else v) for k, v in batch.items()}

        labels = item["input_ids"].clone()
        prompt_length = len(
            self.processor(text=[prompt], images=images or None, return_tensors="pt")[
                "input_ids"
            ][0]
        )
        labels[:prompt_length] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        item["labels"] = labels
        return item


class TimeBudget(_TrainerCallbackBase):  # type: ignore[misc,valid-type]
    """Stop training at a wall-clock deadline, mid-epoch if need be.

    Rented GPU time is a fixed budget, not a variable one. Sizing a run by
    record count means guessing throughput in advance and discovering the guess
    was wrong when the hour runs out with no adapter saved -- and an adapter that
    does not exist scores nothing at all.

    Stopping on the clock inverts that: the run always ends with something to
    benchmark, and how far it got is recorded rather than assumed.
    """

    def __init__(self, hours: float) -> None:
        self.deadline = time.monotonic() + hours * 3600
        self.hours = hours
        self.stopped_early = False

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() >= self.deadline:
            self.stopped_early = True
            control.should_training_stop = True
            print(
                f"\n  time budget of {self.hours:.2f}h reached at step "
                f"{state.global_step}; stopping and saving what exists.",
                flush=True,
            )
        return control


def collate(features: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch

    longest = max(len(f["input_ids"]) for f in features)
    batch: dict[str, Any] = {}

    for key in ("input_ids", "attention_mask", "labels"):
        if key not in features[0]:
            continue
        fill = {"input_ids": pad_token_id, "attention_mask": 0, "labels": -100}[key]
        batch[key] = torch.stack(
            [
                torch.cat(
                    [
                        f[key],
                        torch.full((longest - len(f[key]),), fill, dtype=f[key].dtype),
                    ]
                )
                for f in features
            ]
        )

    # Vision tensors are concatenated rather than stacked: patch counts differ
    # per image, so there is no common leading dimension to stack on.
    for key in ("pixel_values", "image_grid_thw"):
        if key in features[0]:
            batch[key] = torch.cat([f[key] for f in features], dim=0)
    return batch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["a", "b"], required=True)
    ap.add_argument("--data", required=True, type=Path, help="prepared JSONL")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="root the JSONL image paths are relative to (defaults beside --data)",
    )
    ap.add_argument("--resume", type=Path, default=None, help="adapter to continue")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--limit", type=int, default=None, help="cap records, for smoke runs"
    )
    ap.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="checkpoint interval; keep it short on interruptible hardware",
    )
    ap.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="stop training at this wall-clock budget and save what exists",
    )
    ap.add_argument(
        "--s3-checkpoints",
        default=None,
        help="s3:// prefix to stream checkpoints to; spot gives two minutes' notice",
    )
    args = ap.parse_args()

    overrides = {
        key: value
        for key, value in (
            ("epochs", args.epochs),
            ("batch_size", args.batch_size),
            ("learning_rate", args.lr),
            ("max_pixels", args.max_pixels),
            ("seed", args.seed),
        )
        if value is not None
    }
    config = (stage_a if args.stage == "a" else stage_b)(**overrides)
    settings = config.lora

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    from satquery.finetune.targets import (
        LANGUAGE_TARGETS,
        freeze_vision_encoder,
        resolve_targets,
    )

    random.seed(settings.seed)
    torch.manual_seed(settings.seed)

    records = read_jsonl(args.data)
    if args.limit:
        records = records[: args.limit]
    share = evidence_share(records)
    print(f"{config.name}: {len(records):,} records from {args.data}")
    print(f"  evidence preamble on {share:.1%} of records")
    if args.stage == "a" and share < 0.2:
        # Not fatal -- an ablation may deliberately want none -- but it is the
        # setting most likely to be wrong by accident, so it is said loudly.
        print(
            "  WARNING: few records carry a preamble. The adapted model will "
            "meet a prompt shape at inference it barely saw in training. See "
            "docs/DATA.md."
        )

    print(f"loading {BASE_MODEL} @ {BASE_REVISION[:8]} ...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        min_pixels=settings.min_pixels,
        max_pixels=settings.max_pixels,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    frozen = freeze_vision_encoder(model)
    print(f"  froze {frozen} vision-encoder tensors (the projector stays trainable)")

    targets: list[str] = []
    if args.resume:
        print(f"  continuing from {args.resume}")
        model = PeftModel.from_pretrained(model, str(args.resume), is_trainable=True)
    else:
        targets = resolve_targets(
            model,
            include_projector=settings.include_projector,
            include_vision_encoder=settings.include_vision_encoder,
        )
        print(f"  LoRA targets: {', '.join(targets)}")
        # Stated explicitly, because 'we fine-tuned it' is worth nothing
        # without saying which half of the model was actually reached.
        vision = [t for t in targets if t not in LANGUAGE_TARGETS]
        print(f"  vision-side modules adapted: {', '.join(vision) or 'NONE'}")
        model = get_peft_model(
            model,
            LoraConfig(
                r=settings.rank,
                lora_alpha=settings.alpha,
                lora_dropout=settings.dropout,
                target_modules=targets,
                task_type="CAUSAL_LM",
                bias="none",
            ),
        )
    model.print_trainable_parameters()

    image_root = args.image_root or args.data.parent
    dataset = ShareGPTDataset(
        records, processor, image_root, settings.max_sequence_length
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out),
            num_train_epochs=settings.epochs,
            per_device_train_batch_size=settings.batch_size,
            gradient_accumulation_steps=settings.gradient_accumulation,
            learning_rate=settings.learning_rate,
            lr_scheduler_type=settings.scheduler,
            warmup_ratio=settings.warmup_ratio,
            bf16=True,
            logging_steps=10,
            # Steps, not epochs. On rented or preemptible hardware an epoch
            # boundary can be hours away, and a checkpoint that never lands
            # is the same as no training at all.
            save_strategy="steps",
            save_steps=int(args.save_steps),
            save_total_limit=2,
            seed=settings.seed,
            report_to=[],
            remove_unused_columns=False,
            gradient_checkpointing=True,
        ),
        train_dataset=dataset,
        data_collator=lambda f: collate(f, processor.tokenizer.pad_token_id),
    )

    budget = TimeBudget(args.max_hours) if args.max_hours else None
    if budget is not None:
        trainer.add_callback(budget)
        print(f"  stopping after {args.max_hours:.2f}h whatever step it reaches")

    trainer.train()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    processor.save_pretrained(str(args.out))

    # Everything needed to defend or reproduce a number measured against this
    # adapter, beside the adapter rather than in someone's notes.
    metadata = {
        **config.as_dict(),
        "records": len(records),
        "data": str(args.data),
        "evidence_rate": round(share, 4),
        "resumed_from": str(args.resume) if args.resume else None,
        # Which modules were actually reached. The published model card states
        # this, and "we fine-tuned it" means nothing without it.
        "target_modules": list(targets) if not args.resume else None,
        "max_hours": args.max_hours,
        "stopped_on_time_budget": bool(budget and budget.stopped_early),
        "global_step": int(trainer.state.global_step),
        "epochs_completed": round(float(trainer.state.epoch or 0.0), 3),
    }
    (args.out / "satquery_training.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    if args.s3_checkpoints:
        import subprocess

        subprocess.run(
            ["aws", "s3", "sync", str(args.out), args.s3_checkpoints.rstrip("/") + "/"],
            check=False,
        )

    print(f"\nwrote adapter to {args.out}")
    print("serve it with:")
    print(
        f"  python -m vllm.entrypoints.openai.api_server --model {BASE_MODEL} "
        f"--revision {BASE_REVISION} --enable-lora "
        f"--lora-modules qwen3-vl-satquery={args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
