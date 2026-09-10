"""Benchmark dataset acquisition from the sources named in the problem statement.

Each entry records where the data actually comes from, so a reviewer can trace a
score back to an official release rather than to a file someone put on a laptop.

    RSVQA     Zenodo record 6344334, the release linked from rsvqa.sylvainlobry.com
    VRSBench  xiang709/VRSBench on the Hugging Face Hub, the authors' own mirror
    CDVQA     ljx620/CDVQA on the Hub -- see the note on that source below

Downloads are resumable-by-skipping: anything already on disk with the expected
size is left alone, so re-running after an interruption is cheap.
"""

from __future__ import annotations

import json
import re
import shutil
import tarfile
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

ProgressCallback = Callable[["DataProgress"], None]

HF = "https://huggingface.co/datasets"
ZENODO_RSVQA_LR = "https://zenodo.org/api/records/6344334/files"

#: Chunk size for streamed downloads.
_CHUNK = 1 << 20

#: A multi-gigabyte pull over a domestic link breaks often enough that one
#: attempt is not a plan. Retries resume from the partial, so each one keeps
#: whatever the last got.
_DOWNLOAD_ATTEMPTS = 5
_RETRY_BACKOFF = 3.0  # seconds, multiplied by the attempt number


@dataclass
class DataProgress:
    """Snapshot of a dataset download, safe to serialise to a client."""

    state: str = "idle"  # idle | downloading | extracting | ready | error
    dataset: str = ""
    detail: str = ""
    current_file: str = ""
    files_done: int = 0
    files_total: int = 0
    downloaded_bytes: int = 0
    total_bytes: int = 0

    @property
    def percent(self) -> float | None:
        if self.total_bytes <= 0:
            return None
        return round(100.0 * self.downloaded_bytes / self.total_bytes, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "dataset": self.dataset,
            "detail": self.detail,
            "current_file": self.current_file,
            "files_done": self.files_done,
            "files_total": self.files_total,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "percent": self.percent,
        }


@dataclass(frozen=True)
class DataFile:
    """One remote artefact and where it lands."""

    url: str
    dest: str
    size_mb: float = 0.0
    extract: bool = False
    #: Large assets the user opts into rather than pulling by default.
    optional: bool = False


@dataclass(frozen=True)
class DatasetSource:
    """A prescribed benchmark and how to obtain it."""

    name: str
    title: str
    homepage: str
    provenance: str
    root: str
    files: tuple[DataFile, ...]
    #: Marks the dataset usable once present.
    ready_markers: tuple[str, ...] = ()
    note: str = ""
    shard_url: str = ""
    shard_count: int = 0
    shard_size_mb: float = 0.0
    #: Basename stem for downloaded shards. Only cosmetic while each split has
    #: its own root, but a shard called ``test-00000.tar`` sitting in a train
    #: directory is exactly the sort of thing that gets trusted later.
    shard_prefix: str = "test"
    post_process: Callable[[Path, ProgressCallback | None], None] | None = field(
        default=None, repr=False
    )

    def required_files(self, with_optional: bool = False) -> list[DataFile]:
        return [f for f in self.files if with_optional or not f.optional]

    def is_ready(self, data_root: Path) -> bool:
        root = data_root / self.root
        markers = self.ready_markers or tuple(f.dest for f in self.required_files())
        return bool(markers) and all((root / m).exists() for m in markers)

    def describe(self, data_root: Path) -> dict[str, Any]:
        required = self.required_files()
        optional = [f for f in self.files if f.optional]
        return {
            "name": self.name,
            "title": self.title,
            "homepage": self.homepage,
            "provenance": self.provenance,
            "note": self.note,
            "root": str(data_root / self.root),
            "ready": self.is_ready(data_root),
            "download_mb": round(sum(f.size_mb for f in required), 1),
            "optional_mb": round(sum(f.size_mb for f in optional), 1),
            "shards": self.shard_count,
            "shard_size_mb": self.shard_size_mb,
        }


# --------------------------------------------------------------------------
# transfer helpers
# --------------------------------------------------------------------------


def download(
    url: str,
    dest: Path,
    progress: DataProgress,
    on_update: ProgressCallback | None = None,
    attempts: int = _DOWNLOAD_ATTEMPTS,
) -> Path:
    """Stream a URL to disk, resuming a partial transfer rather than restarting.

    The VRSBench imagery is 3.7 GB, and a connection that breaks at 600 MB is
    not unusual on a domestic link. Restarting from zero each time means a large
    file may never complete at all -- the retry is as likely to break as the
    attempt before it. Resuming makes every attempt keep its progress, so the
    download finishes eventually even over a link that cannot hold for an hour.

    The partial is written to ``.part`` and only renamed on success, so an
    interrupted run never leaves a truncated file under the real name where the
    caller's ``final.exists()`` check would take it for a completed download.
    """
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    # Bytes already on disk are progress that was already reported, so the
    # counter starts past them rather than double-counting the resumed prefix.
    already = partial.stat().st_size if partial.exists() else 0
    progress.downloaded_bytes += already

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        have = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}

        try:
            with requests.get(
                url, stream=True, timeout=120, headers=headers
            ) as response:
                # 206 means the server honoured the range and is sending the
                # remainder. A plain 200 means it ignored it and is sending the
                # whole file, so the prefix must be discarded or the two would
                # be concatenated into a corrupt archive.
                resuming = have > 0 and response.status_code == 206
                if have and not resuming:
                    progress.downloaded_bytes -= have
                    have = 0
                response.raise_for_status()

                with partial.open("ab" if resuming else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=_CHUNK):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        progress.downloaded_bytes += len(chunk)
                        if on_update is not None:
                            on_update(progress)

            partial.replace(dest)
            return dest

        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            progress.detail = (
                f"{dest.name}: {type(exc).__name__} after "
                f"{_human_mb(partial.stat().st_size if partial.exists() else 0)}"
                f"; resuming (attempt {attempt + 1}/{attempts})"
            )
            if on_update is not None:
                on_update(progress)
            time.sleep(_RETRY_BACKOFF * attempt)

    raise RuntimeError(
        f"{dest.name}: gave up after {attempts} attempts. "
        f"{_human_mb(partial.stat().st_size if partial.exists() else 0)} is kept "
        f"at {partial.name}, so re-running resumes rather than restarting. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    ) from last_error


def _human_mb(count: int) -> str:
    return f"{count / (1024 * 1024):.0f} MB"


def extract(
    archive: Path,
    into: Path,
    progress: DataProgress,
    on_update: ProgressCallback | None = None,
) -> None:
    """Unpack a zip or tar next to itself, then drop the archive."""
    progress.state = "extracting"
    progress.detail = f"extracting {archive.name}"
    if on_update is not None:
        on_update(progress)

    into.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(into)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(into)
    archive.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# CDVQA: webdataset shards -> the im1/im2 layout the adapter expects
# --------------------------------------------------------------------------


_IMAGE_TURN = re.compile(r"^\s*Image\s+\d+\s*:\s*<image>\s*$", re.IGNORECASE)


def _parse_cdvqa_sample(meta: dict[str, Any]) -> tuple[str, str] | None:
    """Pull the question and answer out of a LLaVA-style conversation record.

    The user turn prefixes the question with one ``Image N: <image>`` line per
    input; those placeholders belong to the packing format, not the question.
    """
    question = answer = ""
    for turn in meta.get("conversations", []):
        value = str(turn.get("value", ""))
        if turn.get("from") == "gpt":
            answer = value.strip()
        else:
            lines = [ln for ln in value.splitlines() if not _IMAGE_TURN.match(ln)]
            question = "\n".join(lines).strip()
    if not question or not answer:
        return None
    return question, answer


def _convert_cdvqa_shards(
    root: Path, on_update: ProgressCallback | None, split: str = "test"
) -> None:
    """Turn downloaded webdataset tars into im1/, im2/ and cdvqa_{split}.json.

    Shards pack each question as ``<key>.0.img``, ``<key>.1.img`` and
    ``<key>.json``. Unpacking to parallel date directories reproduces the
    on-disk layout of the original SECOND-derived release, so the adapter needs
    no special case.

    Several questions share one image pair, so images are written once per
    ``meta.image_id`` rather than once per question -- otherwise a shard of 100
    questions would litter the directory with duplicate copies of the same tile.
    """
    shard_dir = root / "_shards"
    if not shard_dir.is_dir():
        return

    im1, im2 = root / "im1", root / "im2"
    im1.mkdir(parents=True, exist_ok=True)
    im2.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    written: set[str] = set()

    for shard in sorted(shard_dir.glob("*.tar")):
        with tarfile.open(shard) as tf:
            payloads: dict[str, dict[str, bytes]] = {}
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                key, _, suffix = member.name.partition(".")
                data = tf.extractfile(member)
                if data is not None:
                    payloads.setdefault(key, {})[suffix] = data.read()

            for key, parts in sorted(payloads.items()):
                raw, before, after = (
                    parts.get("json"),
                    parts.get("0.img"),
                    parts.get("1.img"),
                )
                if not (raw and before and after):
                    continue
                meta = json.loads(raw)
                parsed = _parse_cdvqa_sample(meta)
                if parsed is None:
                    continue
                question, answer = parsed

                info = meta.get("meta", {})
                name = str(info.get("image_id") or f"{key}.png")
                if name not in written:
                    (im1 / name).write_bytes(before)
                    (im2 / name).write_bytes(after)
                    written.add(name)

                records.append(
                    {
                        "img_name": name,
                        "question": question,
                        "answer": answer,
                        "type": info.get("question_type", ""),
                    }
                )
        shard.unlink(missing_ok=True)

    (root / f"cdvqa_{split}.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )
    shutil.rmtree(shard_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# the prescribed sources
# --------------------------------------------------------------------------

RSVQA_LR = DatasetSource(
    name="rsvqa_lr",
    title="RSVQA Low Resolution",
    homepage="https://rsvqa.sylvainlobry.com/",
    provenance="Zenodo record 6344334 (official release)",
    root="RSVQA",
    files=(
        DataFile(
            f"{ZENODO_RSVQA_LR}/LR_split_test_questions.json/content",
            "LR/LR_split_test_questions.json",
            2.6,
        ),
        DataFile(
            f"{ZENODO_RSVQA_LR}/LR_split_test_answers.json/content",
            "LR/LR_split_test_answers.json",
            1.8,
        ),
        DataFile(
            f"{ZENODO_RSVQA_LR}/LR_split_test_images.json/content",
            "LR/LR_split_test_images.json",
            0.1,
        ),
        DataFile(
            f"{ZENODO_RSVQA_LR}/Images_LR.zip/content",
            "LR/Images_LR.zip",
            90.6,
            extract=True,
        ),
    ),
    ready_markers=(
        "LR/LR_split_test_questions.json",
        "LR/LR_split_test_answers.json",
        "LR/Images_LR",
    ),
)

VRSBENCH = DatasetSource(
    name="vrsbench",
    title="VRSBench (VQA, captioning, referring)",
    homepage="https://vrsbench.github.io/",
    provenance="xiang709/VRSBench on the Hugging Face Hub (authors' mirror)",
    root="VRSBench",
    files=(
        DataFile(
            f"{HF}/xiang709/VRSBench/resolve/main/VRSBench_EVAL_vqa.json",
            "VRSBench_EVAL_vqa.json",
            8.9,
        ),
        DataFile(
            f"{HF}/xiang709/VRSBench/resolve/main/VRSBench_EVAL_Cap.json",
            "VRSBench_EVAL_Cap.json",
            4.6,
        ),
        DataFile(
            f"{HF}/xiang709/VRSBench/resolve/main/VRSBench_EVAL_referring.json",
            "VRSBench_EVAL_referring.json",
            9.8,
        ),
        DataFile(
            f"{HF}/xiang709/VRSBench/resolve/main/Images_val.zip",
            "Images_val.zip",
            3792.4,
            extract=True,
            optional=True,
        ),
    ),
    ready_markers=("VRSBench_EVAL_vqa.json",),
    note=(
        "Annotations are small and download by default. The validation imagery "
        "is 3.8 GB and is opt-in; scoring needs it, inspecting the questions "
        "does not."
    ),
)

CDVQA = DatasetSource(
    name="cdvqa",
    title="CDVQA (change-based VQA)",
    homepage="https://github.com/YZHJessica/CDVQA",
    provenance="ljx620/CDVQA on the Hugging Face Hub",
    root="CDVQA",
    files=(),
    ready_markers=("cdvqa_test.json", "im1", "im2"),
    shard_url=f"{HF}/ljx620/CDVQA/resolve/main/test/test-{{index:05d}}.tar",
    shard_count=397,
    shard_size_mb=76.4,
    note=(
        "The official repository distributes the question file separately from "
        "the SECOND imagery it is built on, behind a manual download. This Hub "
        "mirror packages both together, which is what makes an automated pull "
        "possible. Shards hold 100 samples each; pull as many as you need."
    ),
    post_process=_convert_cdvqa_shards,
)

# --------------------------------------------------------------------------
# the train splits -- Stage B input
# --------------------------------------------------------------------------
#
# Separate sources rather than extra files on the existing ones, and separate
# roots on disk, for one reason: these must never land in the same directory as
# the split they are scored against. `satquery data check-contamination` keys on
# image basename, and mixing the two roots would make a genuine overlap
# indistinguishable from a filesystem accident -- while an unnoticed overwrite
# would replace a test image with its training twin and nothing would report it.
#
# Pull them, then run the guard before training. It is not optional:
#
#   satquery data pull vrsbench_train --with-images
#   satquery data check-contamination data/prepared/stage-b/train.jsonl

VRSBENCH_TRAIN = DatasetSource(
    name="vrsbench_train",
    title="VRSBench train split (VQA, captioning, referring)",
    homepage="https://vrsbench.github.io/",
    provenance="xiang709/VRSBench on the Hugging Face Hub (authors' mirror)",
    root="VRSBench_train",
    files=(
        DataFile(
            f"{HF}/xiang709/VRSBench/resolve/main/VRSBench_train.json",
            "VRSBench_train.json",
            64.9,
        ),
        DataFile(
            f"{HF}/xiang709/VRSBench/resolve/main/Images_train.zip",
            "Images_train.zip",
            8359.3,
            extract=True,
            optional=True,
        ),
    ),
    ready_markers=("VRSBench_train.json",),
    note=(
        "One 65 MB LLaVA-style file, not the three-JSON shape of the EVAL "
        "release: 142,390 conversations over 20,262 images with caption, "
        "referring and VQA interleaved and tagged inline. Read it with the "
        "'vrsbench_train' adapter, never the EVAL adapters. Verified disjoint "
        "from the EVAL images (0 of 20,262 shared). Imagery is ~8 GB and opt-in."
    ),
)

RSVQA_LR_TRAIN = DatasetSource(
    name="rsvqa_lr_train",
    title="RSVQA Low Resolution, train split",
    homepage="https://rsvqa.sylvainlobry.com/",
    provenance="Zenodo record 6344334 (official release)",
    root="RSVQA",
    files=(
        DataFile(
            f"{ZENODO_RSVQA_LR}/LR_split_train_questions.json/content",
            "LR/LR_split_train_questions.json",
            20.9,
        ),
        DataFile(
            f"{ZENODO_RSVQA_LR}/LR_split_train_answers.json/content",
            "LR/LR_split_train_answers.json",
            14.4,
        ),
        DataFile(
            f"{ZENODO_RSVQA_LR}/LR_split_train_images.json/content",
            "LR/LR_split_train_images.json",
            0.5,
        ),
        # The same archive the test split uses -- RSVQA LR ships one image pool
        # and partitions it by id, so this is skipped when it is already on
        # disk rather than fetched twice.
        DataFile(
            f"{ZENODO_RSVQA_LR}/Images_LR.zip/content",
            "LR/Images_LR.zip",
            90.6,
            extract=True,
        ),
    ),
    ready_markers=(
        "LR/LR_split_train_questions.json",
        "LR/LR_split_train_answers.json",
        "LR/Images_LR",
    ),
    note=(
        "Shares root and imagery with 'rsvqa_lr': one image pool of 772 tiles "
        "partitioned by id, and the test split uses only ids 232-331, so only "
        "the question and answer files are genuinely new and Images_LR is "
        "skipped when already present. UNVERIFIED: Zenodo was returning 504 for "
        "every URL in this record -- including the test files that demonstrably "
        "work -- when these entries were written, so the train filenames follow "
        "the release's own convention rather than a checked listing, and the "
        "sizes are estimates. A wrong name fails the pull loudly; a wrong size "
        "only skews the progress bar. Confirm with: satquery data pull "
        "rsvqa_lr_train"
    ),
)

CDVQA_TRAIN = DatasetSource(
    name="cdvqa_train",
    title="CDVQA train split (change-based VQA)",
    homepage="https://github.com/YZHJessica/CDVQA",
    provenance="ljx620/CDVQA on the Hugging Face Hub",
    root="CDVQA_train",
    files=(),
    ready_markers=("cdvqa_train.json", "im1", "im2"),
    shard_url=f"{HF}/ljx620/CDVQA/resolve/main/train/train-{{index:05d}}.tar",
    shard_count=660,
    shard_size_mb=79.1,
    shard_prefix="train",
    note=(
        "65,967 samples across 660 shards of 100. Same webdataset layout and "
        "the same eight question types as the test split. CDVQA is built on "
        "SECOND, which has only a few thousand tile pairs for 122k questions, "
        "so whether train and test share tiles is an open question -- shard 0 "
        "of each is disjoint, but that is 3 images apiece. Run the "
        "contamination guard after pulling and believe what it says; if the "
        "splits do share tiles, this source is not usable for Stage B."
    ),
    post_process=partial(_convert_cdvqa_shards, split="train"),
)

SOURCES: dict[str, DatasetSource] = {
    source.name: source
    for source in (
        RSVQA_LR,
        VRSBENCH,
        CDVQA,
        RSVQA_LR_TRAIN,
        VRSBENCH_TRAIN,
        CDVQA_TRAIN,
    )
}


def get_source(name: str) -> DatasetSource:
    if name not in SOURCES:
        known = ", ".join(sorted(SOURCES))
        raise KeyError(f"unknown dataset '{name}'. Available: {known}")
    return SOURCES[name]


def describe_all(data_root: Path) -> list[dict[str, Any]]:
    return [SOURCES[name].describe(data_root) for name in sorted(SOURCES)]


def pull(
    name: str,
    data_root: Path,
    on_update: ProgressCallback | None = None,
    with_optional: bool = False,
    shards: int = 2,
) -> DataProgress:
    """Fetch one benchmark into ``data_root``.

    Never raises: the returned progress carries the outcome, so a failed pull
    leaves the server up and able to say what went wrong.
    """
    source = get_source(name)
    root = data_root / source.root
    progress = DataProgress(state="downloading", dataset=name)

    def emit() -> None:
        if on_update is not None:
            on_update(progress)

    targets: list[tuple[str, Path, bool]] = [
        (f.url, root / f.dest, f.extract) for f in source.required_files(with_optional)
    ]
    if source.shard_url:
        count = max(1, min(shards, source.shard_count))
        targets += [
            (
                source.shard_url.format(index=i),
                root / "_shards" / f"{source.shard_prefix}-{i:05d}.tar",
                False,
            )
            for i in range(count)
        ]

    progress.files_total = len(targets)
    progress.total_bytes = int(
        sum(f.size_mb for f in source.required_files(with_optional)) * 1024 * 1024
        + (
            source.shard_size_mb * min(shards, source.shard_count) * 1024 * 1024
            if source.shard_url
            else 0
        )
    )
    emit()

    try:
        for url, dest, needs_extract in targets:
            final = dest.with_suffix("") if needs_extract else dest
            if final.exists():
                progress.files_done += 1
                progress.detail = f"{final.name} already present"
                emit()
                continue

            progress.current_file = dest.name
            progress.state = "downloading"
            progress.detail = f"downloading {dest.name}"
            emit()
            download(url, dest, progress, on_update)

            if needs_extract:
                extract(dest, dest.parent, progress, on_update)
            progress.files_done += 1
            emit()

        if source.post_process is not None:
            progress.state = "extracting"
            progress.detail = "assembling image pairs"
            emit()
            source.post_process(root, on_update)

    except Exception as exc:
        progress.state = "error"
        progress.detail = f"{type(exc).__name__}: {exc}"
        emit()
        return progress

    progress.state = "ready"
    progress.detail = f"{source.title} ready at {root}"
    emit()
    return progress


def pull_many(
    names: Sequence[str],
    data_root: Path,
    on_update: ProgressCallback | None = None,
    with_optional: bool = False,
    shards: int = 2,
) -> list[DataProgress]:
    return [pull(n, data_root, on_update, with_optional, shards) for n in names]
