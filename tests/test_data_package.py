"""The ``satquery.data`` public surface, as the call sites actually import it.

This file exists because of a bug the other 217 tests could not see. Adding the
``satquery/data/`` package silently shadowed the sibling ``satquery/data.py``
that held the benchmark downloader, so every ``satquery data list`` and
``satquery data pull`` died with an ImportError -- including the VRSBench imagery
pull that three of the five benchmark configs cannot be scored without.

Nothing caught it because ``api/app.py`` and ``cli.py`` import those names
*inside* their handlers, deferring the failure to runtime on a path CI never
walks. So the assertion here is deliberately about the import surface rather
than behaviour: if a name a call site depends on stops resolving from
``satquery.data``, this fails at commit time instead of in front of a judge.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("pandas")

SRC = Path(__file__).resolve().parents[1] / "src"


def test_downloader_names_resolve_from_the_package():
    """The names ``cli.py`` and ``api/app.py`` import lazily are importable."""
    from satquery.data import SOURCES, DataProgress, describe_all, pull

    assert callable(describe_all)
    assert callable(pull)
    assert set(SOURCES) == {
        "rsvqa_lr",
        "vrsbench",
        "cdvqa",
        "rsvqa_lr_train",
        "vrsbench_train",
        "cdvqa_train",
    }
    assert DataProgress().state == "idle"


def test_train_sources_never_share_a_root_with_the_split_they_are_scored_against():
    """VRSBench and CDVQA train imagery must not land on top of the test tiles.

    Both would otherwise unpack into the same im1/im2 or Images directory as the
    benchmark. That overwrites test tiles with their training twins and, worse,
    makes a genuine train/test overlap indistinguishable from a filesystem
    accident -- the contamination guard keys on image basename.

    RSVQA is the deliberate exception: the release ships one image pool
    partitioned by id, so sharing the root is correct and the guard does the
    real work there.
    """
    from satquery.data import SOURCES

    for train, bench in (("vrsbench_train", "vrsbench"), ("cdvqa_train", "cdvqa")):
        assert SOURCES[train].root != SOURCES[bench].root, train

    assert SOURCES["rsvqa_lr_train"].root == SOURCES["rsvqa_lr"].root


def test_cdvqa_train_writes_its_own_annotation_file():
    """A shared post-process that wrote cdvqa_test.json would overwrite the
    benchmark's own annotations with training questions."""
    from satquery.data import SOURCES

    assert "cdvqa_train.json" in SOURCES["cdvqa_train"].ready_markers
    assert "cdvqa_test.json" in SOURCES["cdvqa"].ready_markers


def test_preparation_names_resolve_from_the_package():
    """Adaptation-side names still resolve alongside the downloader ones."""
    from satquery.data import Record, build_record, prepare, stratified_sample

    assert callable(build_record)
    assert callable(prepare)
    assert callable(stratified_sample)
    assert Record.__slots__


def _deferred_imports(module: Path) -> set[str]:
    """Names imported from ``satquery.data`` inside a function body.

    Walked from the AST rather than by importing: the whole point is that these
    imports do not run until the handler does, which is what let the breakage
    reach a release.
    """
    tree = ast.parse(module.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "satquery.data":
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("relative", ["satquery/cli.py", "satquery/api/app.py"])
def test_lazily_imported_names_exist(relative: str):
    """Every deferred ``from satquery.data import X`` resolves today.

    Catches the shadowing class of failure generally, so a future module added
    beside the package is caught by the same test rather than needing a new one.
    """
    import satquery.data as package

    requested = _deferred_imports(SRC / relative)
    assert requested, f"{relative} no longer defers any satquery.data import"

    missing = sorted(name for name in requested if not hasattr(package, name))
    assert not missing, (
        f"{relative} imports {missing} from satquery.data at runtime, but "
        f"they do not resolve. A module shadowed by a package of the same name "
        f"is the usual cause."
    )
