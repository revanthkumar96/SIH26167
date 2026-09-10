# Sources

Every external dependency this project rests on, with what was verified and when.
Anything stated here was checked against the source, not recalled. Rows marked
**unverified** are assumptions that still need confirming — treat them as risk.

Verified 2026-09-06 unless noted.

---

## Datasets — adaptation

### BigEarthNet.txt

| | |
| --- | --- |
| Hub id | `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` |
| Publisher | TU Berlin / BIFOLD / RSiM |
| Paper | arXiv:2603.29630 — matches the ID the problem statement cites |
| Licence | CDLA-Permissive-1.0 |
| Access | public, **ungated** |
| Payload | one 467 MB parquet, `BigEarthNet.txt.parquet` |
| Content | 464,044 co-registered Sentinel-1 + Sentinel-2 pairs, 9,553,962 text annotations |
| Coverage | Europe only — Finland, Portugal, Serbia, Lithuania, Austria, Ireland, Belgium, Switzerland, Luxembourg, Kosovo |

Columns: `ID, s1_name, patch_id, input, output, type, category, split, latitude,
longitude, country, season, climate_zone`.

| split | rows | | type | rows |
| --- | --- | --- | --- | --- |
| train | 4,674,281 | | binary | 3,625,160 |
| validation | 2,454,690 | | mcq | 3,259,184 |
| test | 2,409,962 | | bounding box | 2,205,686 |
| **bench** | **15,029** | | captioning | 463,932 |

`bench` is a **manually verified** split over 1,082 pairs. It is the only public
cross-modal evaluation set we have found — see `BENCHMARKING.md`.

Categories: `presence, area, count, adjacency, point, reference, relative pos,
season, climate zone, country` (plus `None` for captioning).

**The parquet contains annotations only.** Imagery is separate.

### BigEarthNet v2.0 / reBEN

| | |
| --- | --- |
| Hub id (LMDB mirror) | `hackelle/BigEarthNetV2-LMDB` |
| Original | Zenodo record 10891137 |
| Licence | CDLA-Permissive-1.0 |
| Access | public, ungated |
| Size | **155.4 GB**, one monolithic `BENv2.lmdb/data.mdb` |
| Slice | `hackelle/BigEarthNetV2-Lithuania-Summer-LMDB`, **2.5 GB** |
| Task | multi-label land-cover classification |

`metadata.parquet` columns: `patch_id, labels, split, country, s1_name,
s2v1_name, contains_seasonal_snow, contains_cloud_or_shadow`.

`labels` is a CORINE multi-label list, e.g.
`['Arable land', 'Broad-leaved forest', 'Mixed forest', 'Pastures']`.

**This single download serves both the VLM and the CNN** — imagery for one,
labels for the other. The mirror is community-provided ("unofficial") and
converted with `rico-hdl`; the Zenodo release takes precedence on any conflict.

LMDB layout, confirmed from the publishers' own `ben_txt_datamodule.py`:

- keys are the S2 `patch_id` or the S1 `s1_name`, UTF-8 encoded
- values are safetensors blobs, one array per band
- S2 bands `B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12`
- S1 bands `VV VH`
- bands arrive at native 10/20/60 m, so shapes differ and must be resampled
- patches are **120x120 px** at 10 m — about 1.2 km across

**Sentinel-1 values are already in dB.** The published statistics give VV a mean
of −12.64 and VH −19.35. Do not apply a linear-to-dB conversion.

### Public benchmarks

| Dataset | Role | Split sizes on disk |
| --- | --- | --- |
| RSVQA (LR) | single-image VQA | 10,004 annotations, imagery present |
| VRSBench VQA | single-image VQA | 37,409 annotations |
| VRSBench captioning | scene description | 9,350 annotations |
| VRSBench referring | grounding | 16,159 annotations |
| CDVQA | bi-temporal change VQA | 200 annotations, imagery present |

VRSBench validation imagery is a **3.8 GB opt-in download**; annotations are
23.3 MB and pull by default. As of the last check the three VRSBench splits had
**zero images on disk**, so they cannot be scored until that is pulled.

CDVQA ships as 397 shards of 76.4 MB. RSVQA is a 95.1 MB pull from Zenodo
record 6344334.

### Hidden evaluation set

Held by the organisers: pre-georeferenced, co-registered **Cartosat-2S optical +
RISAT SAR** pairs with undisclosed references. Not obtainable. Two properties
drive design decisions in `TOOLS.md` and `CNN.md`:

- Cartosat-2S is panchromatic (potentially **1 band**) or 4-band MX (B/G/R/NIR).
- With no SWIR band, **NDBI cannot be computed** on that imagery.

---

## Indian-region imagery

| Source | Status | Note |
| --- | --- | --- |
| Microsoft Planetary Computer | **in use** | Sentinel-2 L2A and Sentinel-1 RTC over India, anonymous SAS signing, STAC search, `rendered_preview` and `tilejson` assets per item |
| Bhoonidhi (`bhoonidhi.nrsc.gov.in`) | **unverified** | NRSC portal. Free registration for Indian users. Cartosat/RISAT availability, pricing and licensing not confirmed — register early, do not plan around it |
| Bhuvan | unverified | Visualisation and WMS; download scope unconfirmed |

Planetary Computer is the only Indian-region source confirmed working today and
is what the live feed and the programmatic-supervision plan in `DATA.md` rely on.

---

## Models

| | |
| --- | --- |
| Fine-tune base | `Qwen/Qwen3-VL-2B-Instruct` |
| Pinned revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| Local demo | `qwen3-vl:2b-instruct` via Ollama, 1.9 GB Q4_K_M |
| CNN starting point | BIFOLD publish pretrained reBEN weights — **unverified**, confirm before Phase 4b |

### Trainer support — resolved

Qwen3-VL 2B trains under **plain `transformers` + `peft`**. No LLaMA-Factory or
ms-swift needed. Confirmed from a third-party adapter's published manifest
(`aanandmodi/satquery-qwen3vl-bigearthnet-txt-lora`, Apache-2.0), which trained
on a **Tesla T4** — sm75, 14.56 GiB, no bf16. If it fits there, an A10G is roomy.

Known-good pins, taken from that manifest:

```
transformers==4.57.1
peft==0.17.1
accelerate==1.7.0
qwen-vl-utils==0.0.14
torch 2.10.0+cu128
```

That adapter is **a competitor's work**, not ours. It is cited only as evidence
that the toolchain works. We do not build on it, depend on it, or ship it.

---

## Competitive context

At least a dozen forks of `BigEarthNet.txt` exist on the Hub from other 2026
teams (`Dajyu`, `pluspilot`, `abhixsin`, `Subham05x`, `SohamG2696`,
`trishul2026/*`, `Ignite-2026/*`, and others). The dataset is not a moat.

Where differentiation is actually available, based on what those public artefacts
show: prescribed-split evaluation with prescribed metrics, cross-modal scoring,
bi-temporal capability, evidence-grounded orchestration, and regional adaptation.
See `BENCHMARKING.md` and `FINETUNING.md`.

---

## Runtime

| Component | Version / detail |
| --- | --- |
| Python | 3.11 |
| Ollama | 0.33.2, `POST /api/chat` with `messages[].images` base64 |
| Planetary Computer | STAC API, anonymous SAS signing |
| Leaflet | to be vendored into `static/vendor/`, not CDN-loaded |

Measured on the development laptop (Intel iGPU, Vulkan, no CUDA), single
512 px image, 128-token caption, `qwen3-vl:2b-instruct`:

```
cold: 28.7 s wall | load 12.0 s | prompt 1046 tok @ 110 tok/s | gen 128 tok @ 18.2 tok/s
warm: 33.8 s wall | load  0.1 s | prompt cached           | gen 128 tok @  3.8 tok/s
```

Decode throughput is unstable on that machine (18.2 → 3.8 tok/s on an identical
request) under memory pressure. Ollama caps vision tokens at ~1046 regardless of
whether the input is 512 px or 1024 px, so **input resolution is not a
throughput lever** on this path.
