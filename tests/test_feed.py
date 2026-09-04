"""Live satellite feed: search, empty-result diagnosis, and grid maths.

Network is stubbed. What matters here is that an empty result is explained rather
than shown as a blank grid, and that a pull puts every scene on one shared grid.
"""

import pytest

from satquery import feed


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _feature(item_id, dt, cloud=None, collection="sentinel-2-l2a"):
    return {
        "id": item_id,
        "collection": collection,
        "properties": {"datetime": dt, "eo:cloud_cover": cloud},
        "assets": {
            "rendered_preview": {"href": f"https://example.invalid/{item_id}.png"},
            "tilejson": {"href": f"https://example.invalid/{item_id}.json"},
        },
        "bbox": [72.9, 19.0, 73.0, 19.1],
    }


BBOX = (72.90, 19.02, 73.02, 19.12)


def test_utm_zone_north_and_south():
    assert feed.utm_epsg(72.95, 19.07) == 32643  # UTM 43N, western India
    assert feed.utm_epsg(-58.0, -15.0) == 32721  # UTM 21S, South America


def test_collections_cover_both_modalities():
    modalities = {c["modality"] for c in feed.describe_collections()}
    assert modalities == {"optical", "sar"}


def test_search_maps_stac_assets(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        assert json["collections"] == ["sentinel-2-l2a"]
        return _Response({"features": [_feature("S2_A", "2026-02-06T05:40:21Z", 2.0)]})

    monkeypatch.setattr("requests.post", fake_post)

    scenes = feed.search("sentinel-2-l2a", BBOX)
    assert len(scenes) == 1
    assert scenes[0]["id"] == "S2_A"
    assert scenes[0]["cloud_cover"] == 2.0
    assert scenes[0]["preview"].endswith("S2_A.png")


def test_cloud_filter_is_only_applied_to_optical(monkeypatch):
    seen = {}

    def fake_post(url, json=None, timeout=None):
        seen[json["collections"][0]] = "query" in json
        return _Response({"features": []})

    monkeypatch.setattr("requests.post", fake_post)

    feed.search("sentinel-2-l2a", BBOX, max_cloud=20.0)
    feed.search("sentinel-1-rtc", BBOX, max_cloud=20.0)

    assert seen["sentinel-2-l2a"] is True
    assert seen["sentinel-1-rtc"] is False  # radar has no cloud cover to filter on


def test_empty_optical_result_is_explained_not_blank(monkeypatch):
    """A monsoon footprint returns nothing under a 20% threshold; say why."""
    calls = []

    def fake_post(url, json=None, timeout=None):
        filtered = "query" in json
        calls.append(filtered)
        if filtered:
            return _Response({"features": []})
        return _Response(
            {
                "features": [
                    _feature("S2_A", "2026-08-29T05:40:21Z", 90.0),
                    _feature("S2_B", "2026-08-24T05:40:21Z", 78.0),
                ]
            }
        )

    monkeypatch.setattr("requests.post", fake_post)

    result = feed.search_with_diagnosis("sentinel-2-l2a", BBOX, max_cloud=20.0)

    assert result["scenes"] == []
    assert result["filtered_out"] == 2
    assert "none below 20% cloud" in result["hint"]
    assert "clearest is 78%" in result["hint"]
    # The cloud case is the problem statement's own argument for radar.
    assert "Sentinel-1" in result["hint"]
    assert calls == [True, False]  # filtered first, unfiltered only to diagnose


def test_no_acquisitions_at_all_says_so(monkeypatch):
    monkeypatch.setattr("requests.post", lambda *a, **k: _Response({"features": []}))
    result = feed.search_with_diagnosis("sentinel-2-l2a", BBOX, max_cloud=20.0)
    assert result["filtered_out"] == 0
    assert "No acquisitions at all" in result["hint"]


def test_successful_search_carries_no_hint(monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: _Response(
            {"features": [_feature("S2_A", "2026-02-06T05:40:21Z", 1.0)]}
        ),
    )
    result = feed.search_with_diagnosis("sentinel-2-l2a", BBOX, max_cloud=20.0)
    assert result["hint"] == ""
    assert "filtered_out" not in result


def test_pull_reports_error_instead_of_raising(monkeypatch, tmp_path):
    """A failed pull must leave the server able to explain itself."""

    def boom(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(feed, "_fetch_item", boom)
    monkeypatch.setattr(feed, "target_grid", lambda *a, **k: (None, None))

    progress = feed.pull_scenes(
        [{"id": "S2_A", "collection": "sentinel-2-l2a"}], BBOX, tmp_path
    )
    assert progress.state == "error"
    assert "network unreachable" in progress.detail


def test_progress_percent_needs_a_total():
    assert feed.FeedProgress().percent is None
    assert feed.FeedProgress(bands_done=6, bands_total=12).percent == 50.0


@pytest.mark.parametrize(
    ("collection", "expected_bands"),
    [("sentinel-2-l2a", 12), ("sentinel-1-rtc", 2)],
)
def test_band_plans_match_the_index_tools(collection, expected_bands):
    """12-band S2 is what the optical index tool's band plan expects."""
    bands = feed.S1_BANDS if collection.startswith("sentinel-1") else feed.S2_BANDS
    assert len(bands) == expected_bands
