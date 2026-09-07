"""Dataset ingestion and preparation.

Two concerns, deliberately kept in one package rather than one beside the other:
``benchmarks`` acquires the *evaluation* splits named in the problem statement,
``bigearthnet`` prepares the *adaptation* corpus. Both are "datasets on disk",
both are addressed as ``satquery.data``, and splitting them across a module and a
package of the same name is what broke the downloader once already -- a package
shadows a sibling module, so ``satquery/data.py`` became unreachable the moment
``satquery/data/`` appeared and every ``satquery data`` command died with an
ImportError that no test exercised.
"""

from satquery.data.benchmarks import (
    CDVQA,
    RSVQA_LR,
    SOURCES,
    VRSBENCH,
    DataFile,
    DataProgress,
    DatasetSource,
    describe_all,
    download,
    extract,
    get_source,
    pull,
    pull_many,
)
from satquery.data.bigearthnet import (
    PreparationError,
    Record,
    build_record,
    parse_box,
    prepare,
    redact_caption,
    rewrite_prompt,
    stratified_sample,
    to_milli_box,
    write_jsonl,
)

__all__ = [
    "CDVQA",
    "RSVQA_LR",
    "SOURCES",
    "VRSBENCH",
    "DataFile",
    "DataProgress",
    "DatasetSource",
    "PreparationError",
    "Record",
    "build_record",
    "describe_all",
    "download",
    "extract",
    "get_source",
    "parse_box",
    "prepare",
    "pull",
    "pull_many",
    "redact_caption",
    "rewrite_prompt",
    "stratified_sample",
    "to_milli_box",
    "write_jsonl",
]
