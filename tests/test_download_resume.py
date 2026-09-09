"""Resumable downloads.

Written after a real failure: the 3.7 GB VRSBench imagery broke at 628 MB over a
domestic link. Restarting from zero is not a recovery strategy for a file that
size -- the retry is as likely to break as the attempt before it, so the download
may never complete at all.

The dangerous case is the one that does not raise. If a server ignores the
``Range`` header and replies 200 with the whole file, appending it to the prefix
already on disk produces a corrupt archive of exactly the right shape -- no
error, just an unzip failure much later, or worse, a silently truncated dataset
that scores zero and reads as model failure.
"""

from __future__ import annotations

import pytest

from satquery.data.benchmarks import DataProgress, download

PAYLOAD = b"".join(bytes([i % 251]) for i in range(4096))


class _Response:
    """Enough of ``requests.Response`` for the streaming path."""

    def __init__(self, body: bytes, status: int = 200, fail_after: int | None = None):
        self._body = body
        self.status_code = status
        self._fail_after = fail_after

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")

    #: Chunked on the mock's own terms, not the caller's. download() asks for
    #: 1 MB, which would hand this whole payload over in a single chunk and make
    #: every mid-transfer failure below unreachable -- the tests would pass
    #: without exercising anything.
    CHUNK = 512

    def iter_content(self, chunk_size=None):
        sent = 0
        for start in range(0, len(self._body), self.CHUNK):
            if self._fail_after is not None and sent >= self._fail_after:
                raise ConnectionError("Connection broken: IncompleteRead")
            chunk = self._body[start : start + self.CHUNK]
            sent += len(chunk)
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def install(monkeypatch, responses):
    """Serve ``responses`` in order, recording the headers each request sent."""
    seen: list[dict] = []
    queue = list(responses)

    def fake_get(url, stream=True, timeout=None, headers=None):
        seen.append(dict(headers or {}))
        return queue.pop(0)

    monkeypatch.setattr("requests.get", fake_get)
    return seen


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Retries are tested for behaviour, not for how long they sleep."""
    monkeypatch.setattr("satquery.data.benchmarks.time.sleep", lambda _s: None)


def test_clean_download_writes_the_whole_file(tmp_path, monkeypatch):
    headers = install(monkeypatch, [_Response(PAYLOAD)])
    progress = DataProgress()

    dest = download("http://x/f.zip", tmp_path / "f.zip", progress)

    assert dest.read_bytes() == PAYLOAD
    assert progress.downloaded_bytes == len(PAYLOAD)
    # No prefix on disk, so no Range header is sent.
    assert "Range" not in headers[0]


def test_a_broken_transfer_resumes_from_the_partial(tmp_path, monkeypatch):
    """The failure that prompted this: keep the prefix, ask for the remainder."""
    cut = 2048
    headers = install(
        monkeypatch,
        [
            _Response(PAYLOAD, fail_after=cut),
            _Response(PAYLOAD[cut:], status=206),
        ],
    )
    progress = DataProgress()

    dest = download("http://x/f.zip", tmp_path / "f.zip", progress)

    assert dest.read_bytes() == PAYLOAD
    assert headers[0].get("Range") is None
    assert headers[1]["Range"] == f"bytes={cut}-"


def test_a_server_ignoring_range_restarts_instead_of_corrupting(tmp_path, monkeypatch):
    """The silent one.

    A 200 in reply to a Range request means the whole file is coming. Appending
    it to the prefix already on disk yields a corrupt archive that nothing
    raises about until an unzip fails much later.
    """
    cut = 2048
    install(
        monkeypatch,
        [
            _Response(PAYLOAD, fail_after=cut),
            _Response(PAYLOAD, status=200),  # range ignored
        ],
    )
    progress = DataProgress()

    dest = download("http://x/f.zip", tmp_path / "f.zip", progress)

    assert dest.read_bytes() == PAYLOAD, "prefix must be discarded, not appended to"
    assert len(dest.read_bytes()) == len(PAYLOAD)


def test_progress_does_not_double_count_a_resumed_prefix(tmp_path, monkeypatch):
    """A resumed run must not report more bytes than the file contains."""
    partial = tmp_path / "f.zip.part"
    partial.write_bytes(PAYLOAD[:1024])
    install(monkeypatch, [_Response(PAYLOAD[1024:], status=206)])

    progress = DataProgress()
    download("http://x/f.zip", tmp_path / "f.zip", progress)

    assert progress.downloaded_bytes == len(PAYLOAD)


def test_the_real_name_never_holds_a_truncated_file(tmp_path, monkeypatch):
    """The caller skips a file that already exists, so a truncated one is poison."""
    install(monkeypatch, [_Response(PAYLOAD, fail_after=1024)] * 5)
    dest = tmp_path / "f.zip"

    with pytest.raises(RuntimeError):
        download("http://x/f.zip", dest, DataProgress(), attempts=5)

    assert not dest.exists(), "a failed download must not leave the final name"
    assert (tmp_path / "f.zip.part").exists(), "the prefix is kept, to resume from"


def test_giving_up_says_what_is_kept_and_that_rerunning_resumes(tmp_path, monkeypatch):
    install(monkeypatch, [_Response(PAYLOAD, fail_after=1024)] * 3)

    with pytest.raises(RuntimeError) as excinfo:
        download("http://x/f.zip", tmp_path / "f.zip", DataProgress(), attempts=3)

    message = str(excinfo.value)
    assert "resumes rather than restarting" in message
    assert "ConnectionError" in message


def test_retries_stop_at_the_limit(tmp_path, monkeypatch):
    headers = install(monkeypatch, [_Response(PAYLOAD, fail_after=512)] * 4)

    with pytest.raises(RuntimeError):
        download("http://x/f.zip", tmp_path / "f.zip", DataProgress(), attempts=4)

    assert len(headers) == 4


def test_each_attempt_keeps_the_ground_the_last_one_gained(tmp_path, monkeypatch):
    """The property that makes a flaky 3.7 GB pull finish at all."""
    headers = install(
        monkeypatch,
        [
            _Response(PAYLOAD, fail_after=1024),
            _Response(PAYLOAD[1024:], status=206, fail_after=1024),
            _Response(PAYLOAD[2048:], status=206),
        ],
    )
    progress = DataProgress()

    dest = download("http://x/f.zip", tmp_path / "f.zip", progress)

    assert dest.read_bytes() == PAYLOAD
    assert [h.get("Range") for h in headers] == [None, "bytes=1024-", "bytes=2048-"]
