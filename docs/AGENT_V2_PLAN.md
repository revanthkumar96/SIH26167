# Agent v2 — application rebuild and the core ML programme

Branch: `agent_v2`. Supersedes the Kaggle-scoped schedule in `ML_PLAN.md`, which
assumed free T4/P100 time; the compute base is now AWS with a $100 credit grant.

Two things change at once, and they are related. The application commits to a
single model family served over HTTP instead of a local Ollama process, and the
model itself stops being a stock checkpoint. The second is not optional: the
problem statement lists "a generic LLM/VLM with no remote-sensing fine-tuning or
domain adaptation" under **Constraints and disqualifiers**. Everything in Part B
exists to clear that bar.

---

## Part A — Application changes

### A1. The map becomes the input surface

Today the UI is four tabs of forms and the imagery is a thumbnail. In v2 a
full-screen slippy map carries Sentinel-1 and Sentinel-2 tiles, and the query
interface is a floating widget over it.

The point is not decoration. **If the map is only a backdrop, it is worse than
what we have** — it costs screen space and adds nothing. The map earns its place
by becoming the way analysis inputs are chosen:

1. Pan/zoom to an area of interest.
2. Scrub a date control to pick scenes; the optical and SAR layers update.
3. Draw a box. That rectangle is the analysis window.
4. The existing `/api/feed/load` pulls co-registered COG windows for exactly that
   box, onto a shared UTM grid, and hands them to the controller.
5. The trace streams into the chat widget as it runs; evidence overlays (water
   mask, change mask, grounding box) render back onto the map as layers.

Step 5 is the part that makes it a remote-sensing tool rather than a chat app.
A grounding box drawn as a map layer over the scene it came from is
self-evidently georeferenced evidence; the same box in a side panel is a picture.

**Tiles.** No tile server needed. `feed.py` already surfaces the per-item
`tilejson` asset from Planetary Computer (`src/satquery/feed.py:169`), which is
an XYZ endpoint we can hand straight to a raster layer. The deliberate decision
recorded at `src/satquery/feed.py:7` — not to stand up TiTiler — still holds.

**Library.** Leaflet, vendored into `static/vendor/`, not loaded from a CDN.
We only need raster XYZ layers, which is Leaflet's simplest case, and it is
~145 KB. Vendoring is a hackathon-venue decision: **the demo must work on bad
wifi or none.** A CDN `<script>` tag is a single point of failure on stage.

**Optical/SAR comparison.** A swipe divider between the two layers, not a
toggle. Side-by-side comparison of the same ground at the same instant is how
the complementarity argument gets made visually — water is dark in SAR and the
optical scene shows why. This is also how analysts actually look at these pairs.

Offline fallback: when no network is available the map degrades to the bundled
GeoTIFF samples rendered as image overlays on their real bounding boxes, so the
same UI demonstrates the same flows without Planetary Computer.

### A2. Catalogue trimmed to one model family

The bake-off is over. `MODEL_CATALOG` drops `qwen25-vl-7b-ollama`,
`qwen25-vl-3b`, `internvl3-2b-hf`, `qwen3-vl-4b` and `qwen3-vl-8b`, leaving two
entries that matter:

| id | what it is | role |
| --- | --- | --- |
| `qwen3-vl-base` | stock Qwen3-VL weights | the "before" column |
| `qwen3-vl-satquery` | same weights + our LoRA adapter | the "after" column |

The benchmark matrix keeps its multi-model axis, because that axis is now
carrying the argument the judges care about: base versus adapted, same harness,
same prompts, same splits. That comparison is the evidence that requirement 1 of
the mandatory functional scope was met.

### A3. Serving moves to AWS

**Recommended: EC2 `g5.xlarge` (A10G, 24 GB) running vLLM's OpenAI-compatible
server.** Reasons, in order:

- vLLM serves LoRA adapters directly (`--enable-lora`), so base and adapted
  models are two `model` names against one process — exactly the matrix's shape.
- `build_messages()` in `eval/backends/base.py:45` already emits OpenAI-format
  content with base64 `image_url` parts. The remote backend is close to free.
- A10G is Ampere, so **bf16 is available** — unlike the Kaggle T4/P100 path in
  `ML_PLAN.md`, which forced fp16 plus GradScaler. One less source of instability.
- Same instance type trains and serves, so there is one environment to build.

Rejected: SageMaker real-time endpoints (more per-hour cost and more moving
parts for no benefit at this scale); Bedrock (does not host custom Qwen weights).

New backend `openai_compat`, configured by `SATQUERY_VLM_BASE_URL` and
`SATQUERY_VLM_API_KEY`, joining `BACKENDS`. It talks to anything OpenAI-shaped,
which means the same code path works against a local vLLM, the EC2 box, or a
teammate's tunnel.

**Keep the Ollama backend.** Not as the default — as the offline fallback. Demo
day with no network and a dead EC2 instance is a real failure mode, and a 1.9 GB
local model that answers in ~20 s is a far better outcome than a stack trace.
This costs us nothing: the backend already exists and is tested.

**Cost control is a first-class requirement, not an afterthought.** With $100:

| item | rate | budget |
| --- | --- | --- |
| `g5.xlarge` on-demand | ~$1.01/hr | fallback only |
| `g5.xlarge` spot | ~$0.30–0.40/hr | primary |
| S3 storage + requests | — | ~$3 |
| EBS gp3 (200 GB, persistent) | ~$16/month | prorated ~$8 |

Non-negotiable guardrails, because the single likeliest way to lose this budget
is a forgotten running instance:

- AWS Budgets alarm at $40 and $75, email + hard stop review.
- Idle auto-shutdown: a systemd timer that halts the box after 30 min with no
  GPU utilisation and no active SSH session.
- Checkpoints stream to S3 during training, never only to EBS — spot instances
  get reclaimed with two minutes' notice.
- Region chosen for spot A10G availability (`us-east-1` / `us-west-2` are
  deeper than `ap-south-1`); accept the latency, we are not serving users.

---

## Part B — Core ML phases

Eight phases. Effort is person-hours; cost is AWS spend.

### Phase 0 — Infrastructure

| | |
| --- | --- |
| **Deliverable** | A GPU box we can start, train on, serve from, and stop safely. |
| **Work** | AWS account hardening, IAM role, S3 bucket, security group, spot request, Deep Learning AMI (PyTorch 2.x / CUDA 12.x), persistent EBS, idle-shutdown timer, budget alarms. |
| **Effort** | 4 h |
| **Cost** | ~$2 |
| **Exit** | `nvidia-smi` shows the A10G; a checkpoint round-trips to S3; the idle timer demonstrably halts the box. |

### Phase 1 — Data acquisition and preparation

The most underestimated phase in the whole programme. Budget accordingly.

**1a. BigEarthNet.txt** — verified available and usable.
`BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` on the Hub (TU Berlin / BIFOLD / RSiM),
public, **ungated**, **CDLA-Permissive-1.0**, arXiv 2603.29630 — matching the ID
the problem statement cites. Phase 1a is unblocked.

What it actually contains: **464,044 co-registered Sentinel-1 + Sentinel-2 pairs
over Europe** with **9,553,962 text annotations** in a single 467 MB parquet.

| split | rows | | type | rows |
| --- | --- | --- | --- | --- |
| train | 4,674,281 | | binary | 3,625,160 |
| validation | 2,454,690 | | mcq | 3,259,184 |
| test | 2,409,962 | | bounding box | 2,205,686 |
| **bench** | **15,029** | | captioning | 463,932 |

Columns: `ID, s1_name, patch_id, input, output, type, category, split, latitude,
longitude, country, season, climate_zone`. Categories span presence, area, count,
adjacency, point, reference, relative position, season, climate zone, country.

Four consequences, all of which change the plan:

**(i) The parquet is annotations only.** Imagery is a separate download of
BigEarthNet v2.0 from bigearth.net, then a conversion pass with `rico-hdl`
(a Rust tool from the same group) into `safetensors`-in-LMDB. So Phase 1a is a
three-step pipeline — parquet, imagery, LMDB encode — not a one-line
`load_dataset`. Budget EBS and time for the imagery tranche accordingly, and
verify its true size before provisioning the volume.

**(ii) The `bench` split solves a problem we had already hit.** 15,029
annotations over 1,082 pairs, *manually verified*, cross-modal. When the
cross-modal prompt was rewritten, the note in `eval/prompts.py:61` recorded that
**no public benchmark covers cross-modal analysis** — RSVQA and VRSBench are
single-image, CDVQA is bi-temporal — so that path could only be demoed, never
scored. This split closes exactly that gap. Add it to `configs/bench/` as a
first-class benchmark and the mandatory cross-modal criterion becomes measurable
instead of merely demonstrable.

**(iii) 2.2M bounding-box rows feed grounding directly** — but the dataset's own
loader takes `point_token` and `ref_token` parameters, implying a
`<ref>…</ref><point>…</point>` output convention. Ours is `[x1,y1,x2,y2]` on a
0–1000 grid (`eval/prompts.py:38`). Pick one convention and convert at
preparation time; a mismatch here silently scores zero on every grounding item.

**(iv) The geography is Europe only** — Finland, Portugal, Serbia, Lithuania,
Austria, Ireland, Belgium, Switzerland, Luxembourg, Kosovo. **There is not one
Indian scene in it.** The hidden ISRO set is Indian terrain seen by Cartosat-2S
and RISAT. Boreal Finland and Mediterranean Portugal do not teach Indian
land-cover priors, so BigEarthNet.txt buys us *sensor* and *cross-modal*
adaptation, not *regional* adaptation. That gap has to be filled from elsewhere
— VRSBench for resolution, and optionally instruction data we generate over the
Indian Sentinel scenes the live feed already pulls.

Sampling: draw a subset from the `train` split balanced across `type`, roughly
100–200k rows over 30–50k unique pairs. Patches are small (120x120, 12-band S2
plus 2-band S1), so the image tranche stays in the tens of GB.

**1b. VRSBench train split** — ~29k high-resolution (~0.3 m) images with
captions, VQA and referring expressions. This is the *resolution* match to
Cartosat-2S, where BigEarthNet is the *modality* match. Both are needed.

**1c. RSVQA train split** — Sentinel-2-scale VQA, matching the LR benchmark.

**1d. CDVQA train split** — bi-temporal change VQA, built on SECOND imagery.

**1e. Unify into one instruction schema.** All four become ShareGPT-style JSONL
so a single trainer consumes them:

```json
{"images": ["s2/xxx.png", "s1/xxx.png"],
 "conversations": [{"from": "human", "value": "<image><image>..."},
                   {"from": "gpt", "value": "..."}]}
```

**1f. Preprocessing must mirror serving exactly.** This is where train/serve
skew gets introduced and then costs a week to find. The rendering the trainer
sees has to be the rendering `eval/images.py` produces at inference:

- **S2**: band selection (true colour B04/B03/B02; false colour NIR for a second
  view), percentile stretch, uint8.
- **S1**: linear power → dB — the conversion already in the codebase, added
  after RTC gamma0 was misread as 94% water. Then a speckle filter (Refined Lee),
  which we do **not** yet have and which real practice treats as mandatory before
  any thresholding. Then VV/VH/(VV−VH) as a pseudo-RGB, the standard SAR
  visualisation.
- One normalisation module, imported by both paths. Not two implementations that
  are "the same".

| | |
| --- | --- |
| **Effort** | 20–25 h |
| **Cost** | ~$5 (S3 + egress) |
| **Exit** | One JSONL per split, a rendering smoke test that asserts trainer and server produce byte-identical tensors for the same scene. |

### Phase 2 — Baseline benchmark (before)

Serve stock Qwen3-VL on the EC2 box and run all five configs on the prescribed
test splits, through the existing harness.

This is not a formality. It produces the "before" column, it proves the remote
backend works end to end, and it surfaces prompt/format problems while they are
still cheap — the truncation and terse-crossmodal defects found on the local
model are exactly the class of thing this catches.

| | |
| --- | --- |
| **Effort** | 4 h |
| **Cost** | ~$5 (3–5 GPU h) |
| **Exit** | A full result matrix, committed, with OA/AA, BLEU/METEOR/ROUGE-L/CIDEr, Acc@IoU0.5/mIoU. |

### Phase 3 — Adaptation strategy

Verify first, and before anything else: **does the chosen trainer support
Qwen3-VL?** Qwen2.5-VL has deep, well-trodden LoRA support in LLaMA-Factory,
ms-swift and unsloth; Qwen3-VL is newer. If support is thin, the correct move is
to fine-tune Qwen2.5-VL 3B and keep Qwen3-VL as the unadapted fast path — the
registry already carries a per-tool `adapter` field, so the architecture absorbs
this. Do not discover this in Phase 4.

Two stages, both LoRA:

**Stage A — domain and cross-modal adaptation, on BigEarthNet.txt.** Teaches SAR
vocabulary, backscatter intuition, and joint optical+SAR reasoning. This is the
stage that literally satisfies mandatory scope item 1 using the dataset the
problem statement names.

**Stage B — instruction tuning on the task mixture** (VRSBench + RSVQA + CDVQA
train splits). Teaches the *output formats the metrics reward*: terse answers
for exact-match VQA, `[x1,y1,x2,y2]` for grounding, one or two sentences for
captioning. A model that knows the right answer but phrases it wrongly scores
zero on exact match, so this stage is worth more raw points than Stage A.

Starting configuration:

| knob | value | why |
| --- | --- | --- |
| LoRA rank / alpha | 32 / 64 | headroom for a genuine domain shift |
| dropout | 0.05 | standard |
| target modules | `q,k,v,o,gate,up,down` **+ the vision–language projector** | the projector is where optical/SAR domain shift actually lands |
| vision encoder | frozen | full ViT tuning needs far more data and invites forgetting |
| lr | 1e-4, cosine, 3% warmup | conventional for LoRA |
| precision | **bf16** | A10G is Ampere; no GradScaler needed |
| effective batch | 64–128 via grad accumulation | |
| epochs | 1–2 | VLM instruction tuning overfits fast |
| `max_pixels` | tuned down | vision tokens dominate memory; this is *the* OOM lever |

| | |
| --- | --- |
| **Effort** | 8 h |
| **Cost** | ~$3 (short probe runs) |
| **Exit** | Trainer support confirmed in writing; a 200-step run completes and loss descends. |

### Phase 4 — Training runs

Stage A then Stage B, checkpointing to S3 throughout.

| stage | samples | est. GPU h |
| --- | --- | --- |
| A — BigEarthNet.txt, 1 epoch | ~50k | 4–7 |
| B — task mixture, 1–2 epochs | ~60–80k | 6–12 |

| | |
| --- | --- |
| **Effort** | 6 h attended + wall-clock |
| **Cost** | $10–20 (spot) |
| **Exit** | Adapter artefacts in S3, training curves recorded, run config committed. |

### Phase 5 — Adapted benchmark (after)

Identical harness, adapter loaded via vLLM `--lora-modules`. Base and adapted
appear as two rows of the same matrix.

**This comparison is the single most important artefact of the whole project.**
It is what converts "we used an off-the-shelf model" into "we adapted a model
and here is the measured delta", which is the difference between a disqualified
entry and a compliant one.

| | |
| --- | --- |
| **Effort** | 3 h |
| **Cost** | ~$5 |
| **Exit** | Before/after matrix rendered in the Benchmarks tab. |

### Phase 6 — Ablation, per-task adapters, normalised scoring

- Determine which stage helped which task. If Stage B helps grounding but hurts
  captioning, split into per-task adapters — `VLMTool` already names an
  `adapter` per tool, so the controller needs no change.
- **Implement score normalisation.** The problem statement requires heterogeneous
  metrics to be normalised before aggregation, and we do not do this today. It is
  a scored criterion we are currently forfeiting.

| | |
| --- | --- |
| **Effort** | 8 h |
| **Cost** | ~$8 |

### Phase 7 — Integration

Adapter plumbing through `BackendConfig` into the serving path; catalogue
reduced to the two entries; matrix wired to the remote backend; demo-day runbook
with instance start/stop and the offline fallback rehearsed.

| | |
| --- | --- |
| **Effort** | 10 h |
| **Cost** | ~$5 |

### Phase 8 — Hidden-set robustness

The ISRO/SAC set is **Cartosat-2S optical + RISAT SAR**, and it will not look
like Sentinel-2. Two concrete consequences that need code, not hope:

**Band count.** Cartosat-2S is panchromatic — potentially a **single band** —
or 4-band multispectral (B/G/R/NIR) for the MX payload. Our optical path assumes
a 12-band Sentinel-2 stack. A 1-band PAN GeoTIFF must not crash `preview_bands()`
or `optical_indices`; it must report `applicable: false` cleanly and let the run
continue.

**Index availability.** With 4-band MX there is a NIR band, so **NDWI works**.
There is no SWIR, so **NDBI does not**. Built-up therefore cannot come from the
optical index on the hidden set — it has to come from SAR backscatter.

That is not a limitation to apologise for. It is precisely the complementarity
the problem statement is testing, and our architecture already answers it: when
`optical_indices` reports NDBI unavailable, `sar_indices` still supplies the
built-up fraction, and the cross-modal prompt cites which sensor supports which
finding. **Build a synthetic Cartosat-like fixture — 1-band and 4-band, sub-metre
GSD — and add it to the test suite**, so this path is proven rather than assumed.

| | |
| --- | --- |
| **Effort** | 10 h |
| **Cost** | ~$3 |

---

## Totals

| phase | effort | cost |
| --- | --- | --- |
| 0 Infrastructure | 4 h | $2 |
| 1 Data | 20–25 h | $5 |
| 2 Baseline benchmark | 4 h | $5 |
| 3 Strategy | 8 h | $3 |
| 4 Training | 6 h + wall-clock | $10–20 |
| 5 Adapted benchmark | 3 h | $5 |
| 6 Ablation + normalisation | 8 h | $8 |
| 7 Integration | 10 h | $5 |
| 8 Hidden-set robustness | 10 h | $3 |
| **Total** | **~75–80 h** | **~$46–56** |

Roughly half the credit grant, leaving genuine headroom for a failed spot run,
a second training pass after ablation, and demo-day serving.

## Critical path

**Phase 1a is resolved.** BigEarthNet.txt is public, permissively licensed and
carries everything the adaptation stage needs. What remains of that phase is
engineering — pulling the v2.0 imagery and encoding it — not a risk.

**Phase 3's trainer-support check is now the only open gate.** Whether
LLaMA-Factory / ms-swift support Qwen3-VL, or only Qwen2.5-VL, decides the
fine-tune target and therefore Phase 4. It is cheap to answer and expensive to
discover late, so answer it next.

The residual risk is no longer availability but **domain distance**: Europe-only
training data against an Indian evaluation set, and 120x120 patches against
sub-metre Cartosat imagery. Both are mitigated by the mixture in 1b–1d rather
than by anything in 1a.

## Order of work

1. Phase 0 and Phase 1a in parallel — infrastructure while the dataset question
   is settled.
2. Phase 2 baseline as soon as the box serves, because it de-risks the harness.
3. Phase 1b–1f while Phase 2 runs.
4. Phase 3 verification, then Phase 4.
5. Phase 5, then 6 and 8 in parallel, then 7.

Application work in Part A is independent of Phases 1–6 and can proceed
alongside; only A3's remote backend is shared, and it is needed by Phase 2.
