# SatQuery AI — ML Training Plan (Kaggle free tier)

Ordered plan for the ML core: datasets → pretraining → fine-tuning → evaluation,
sized to **Kaggle free tier: 30 GPU hours/week, 12 h max per session**.

Planning envelope: **8 weeks**, ~240 GPU hours total. Weeks are relative — adjust
once the finale date is fixed.

---

## 0. Hardware reality and what it forces

| Constraint | Consequence |
|---|---|
| 30 GPU h/week, **does not roll over** | Under-using week 1 is capacity burned forever. Front-load, never back-load. |
| 12 h hard session cap | Every training script **must** checkpoint + resume. Non-negotiable, built in Stage 0. |
| P100 is sm60 | `bitsandbytes` 4-bit needs sm75+. **Always select `GPU T4 x2`**, never P100, for QLoRA. |
| T4 is Turing | **No bf16.** fp16 + GradScaler only. No FlashAttention-2 (needs Ampere) → use SDPA. |
| 16 GB per T4 | 7B VLM fp16 weights alone are ~15 GB. Training does not fit. |
| ~20 GB `/kaggle/working` | Prune checkpoints; push adapters to a Kaggle Dataset, not the working dir. |
| Internet off by default | Enable the toggle (needs phone verification) or vendor weights as a Dataset. |

### Revised model choice

**Base: `Qwen2.5-VL-3B-Instruct` + QLoRA (NF4), not 7B.**

The earlier 7B recommendation assumed a 24 GB card. On 2×T4 the 3B is the honest
choice: it trains ~2.5× faster, fits with headroom for multi-image (pair) samples, and
lets us train **four** adapters inside the budget instead of one. Keep 7B as a stretch
goal if an external GPU appears — the code path is identical, only the config changes.

Vision tokens dominate cost in Qwen2.5-VL. Cap `min_pixels` / `max_pixels` explicitly
per adapter; pair tasks (change, cross-modal) get half the per-image budget of
single-image tasks.

---

## 1. GPU hour budget

| Stage | Work | GPU h | Week |
|---|---|---|---|
| S0 | Env, smoke configs, checkpoint/resume harness | 3 | W1 |
| E0 | Zero-shot baseline eval, all 4 benchmarks | 8 | W1 |
| C | Change-mask CNN (LEVIR-CD) | 12 | W1 |
| A | Cross-modal encoder pretrain (BigEarthNet S1+S2) | 25 | W2 |
| B1 | VQA adapter (RSVQA + VRSBench VQA) | 25 | W3 |
| B2 | Caption adapter (VRSBench + BEN.txt) | 15 | W4 |
| B3 | Change-VQA adapter (CDVQA) | 22 | W5 |
| B4 | Cross-modal adapter (BEN.txt S1+S2 pairs) | 20 | W6 |
| B* | Best-config re-runs, longer schedules | 25 | W7 |
| EF | Final eval sweep + freeze | 15 | W8 |
| — | Per-stage eval + reserve | ~45 | spread |
| | **Total** | **~215 / 240** | |

~10 % headroom. The reserve is real — first runs fail.

---

## 2. Ordered stages

### Stage 0 — Foundations (W1) · 2 person-days · 3 GPU h

Scaffolding every later stage depends on.

- `src/satquery/ml/` package: `config`, `data`, `models`, `train`, `eval`.
- **Checkpoint/resume harness** — save every ~30 min (step-indexed: model, optimizer,
  scheduler, RNG state), auto-detect and resume from a Kaggle Dataset. Without this the
  12 h cap makes anything longer than one session impossible.
- **Smoke config** per training script: 50 steps, tiny subset, runs in <10 min.
- Kaggle runner using **Save Version → Run All** (headless batch, survives a closed
  browser, gets the full 12 h).

**Done when:** a smoke run trains, is killed mid-run, and resumes to identical loss.

---

### Stage D — Data preparation (W1–W2) · 4 person-days · **0 GPU h**

All of this runs on **CPU sessions**, which do not consume GPU quota. Doing image
decode or resize inside a GPU session is quota set on fire.

| Dataset | Role | Prep output |
|---|---|---|
| BigEarthNet.txt (S1+S2) | Mandatory RS adaptation | Sharded parquet/WebDataset: 12-band S2 + 2-band S1 + labels + text |
| VRSBench | Caption / VQA / grounding | Chat-formatted JSONL + resized images |
| RSVQA (LR + HR) | Single-image VQA | Chat-formatted JSONL, per-type tags kept |
| CDVQA | Change VQA | Bi-temporal pairs + chat JSONL |
| LEVIR-CD / S2Looking | Change masks | 256×256 crops, A/B/label |

Two rendering paths, decided here:

- **Raw multiband** (12-band S2, 2-band SAR) → the cross-modal encoder (Stage A).
- **3-channel renderings** → the VLM. S2 → true colour; SAR → `[VV, VH, VV/VH]` in dB.
  The VLM cannot eat 12 bands; this is the bridge.

Publish each prepared corpus as a **private Kaggle Dataset** — versioned, attaches
read-only, zero prep cost in every later session.

**Open item:** BigEarthNet.txt's annotation schema is unverified (arXiv:2603.29630).
The loader is written adapter-style against a documented expected schema, so only one
file changes once we confirm the real format.

**Done when:** every loader passes a validation script (counts, splits, no missing
files, sane label distribution) and one batch renders correctly to disk.

---

### Stage E0 — Zero-shot baseline (W1) · 1 person-day · 8 GPU h

Run the **untuned** base model over all four benchmark test splits before any training.

Not busywork: without a baseline you cannot prove adaptation helped, and "we fine-tuned
it" with no measured delta is exactly what the problem statement calls out as
insufficient. It also shakes out the eval harness while stakes are low.

**Done when:** a numbers table exists for VQA accuracy, caption BLEU/METEOR/ROUGE-L/CIDEr,
CDVQA OA/AA, and grounding Acc@0.5 — reproducible from one command.

---

### Stage C — Change-mask detector (W1) · 2 person-days · 12 GPU h

Small Siamese CNN (ConvNeXt-T or ResNet-18 encoder, difference decoder, BCE + Dice) on
LEVIR-CD. Independent of the VLM track, so it runs while data prep is still going and
soaks up W1 quota that would otherwise expire.

Feeds two things: the change-map IoU/F1 metric the problem statement mentions, and
spatial context injected into the Stage B3 change adapter's prompt.

**Done when:** IoU ≥ 0.75 on LEVIR-CD test, exported to ONNX, runs on CPU in <2 s.

---

### Stage A — Cross-modal encoder pretraining (W2) · 3 person-days · 25 GPU h

**This is the mandatory "remote-sensing adaptation" deliverable.** Ties directly to
functional requirements #1 and #4.

Two towers over co-registered BigEarthNet S1/S2:

- Optical tower: 12-band stem → ViT-S / ConvNeXt-T
- SAR tower: 2-band stem → same backbone, separate weights
- Loss: InfoNCE between S1↔S2 embeddings of the same patch **+** BCE multi-label head
  (BigEarthNet-19) on the fused embedding.

CROMA/DeCUR-style in spirit, cut to fit the budget. Initialise from public CROMA or
DeCUR weights if we can vendor them — saves ~10 GPU h and lifts the ceiling.

Output: the fusion encoder behind the optical–SAR specialist tool, plus a defensible
"we adapted a visual component on BigEarthNet" claim with a metric attached.

**Done when:** multi-label mAP clearly beats an ImageNet-init baseline, and S1↔S2
retrieval top-1 is well above chance.

---

### Stage B — VLM adapters (W3–W6) · 8 person-days · 82 GPU h

Four QLoRA adapters on one frozen 3B base (r=16, α=32, attention + MLP projections).
Serving multiple LoRAs on one base is what makes the "specialist model registry"
affordable — and each adapter swap is a legible line in the execution trace.

| # | Adapter | Train data | Week | GPU h |
|---|---|---|---|---|
| B1 | `vqa` | RSVQA train + VRSBench VQA train | W3 | 25 |
| B2 | `caption` | VRSBench caption train + BEN.txt captions | W4 | 15 |
| B3 | `change` | CDVQA train (+ change captioning) | W5 | 22 |
| B4 | `crossmodal` | BEN.txt S1+S2 pairs | W6 | 20 |

Order is deliberate: B1 first because VQA is the mandatory scored task and its loss and
data format shake out the whole SFT path; B4 last because it depends on Stage A's
renderings and is the least benchmark-constrained.

Per adapter: smoke → one short run → eval → one tuned run → eval → freeze. Two shots
each. Resist a third; the quota is better spent on the next adapter.

Grounding trains as a **secondary head on B1** (VRSBench referring split), but
**captioning is the declared scored task** — see strategy notes.

**Done when:** each adapter beats the E0 zero-shot baseline on its own benchmark, and
the delta is written into the results table.

---

### Stage EF — Final evaluation and freeze (W8) · 2 person-days · 15 GPU h

Full sweep on prescribed test splits, per-question-type breakdowns, normalised
aggregate mirroring the judging table, model card, reproduce-from-scratch script.

**Done when:** `make eval` reproduces every number in the submission from frozen
checkpoints.

---

## 3. Weekly GPU allocation

| Week | Stages | GPU h |
|---|---|---|
| W1 | S0 + E0 + C (data prep runs on CPU) | 23 |
| W2 | A | 25 |
| W3 | B1 + eval | 28 |
| W4 | B2 + B1 refinement + eval | 27 |
| W5 | B3 + eval | 27 |
| W6 | B4 + eval | 26 |
| W7 | Best-config re-runs | 25 |
| W8 | EF + reserve | 25 |

No week exceeds 30. Every week uses most of its quota, because unused quota is gone.

---

## 4. Standing rules

1. **CPU sessions for anything that is not a backward pass** — prep, scoring cached
   generations, dataset packaging.
2. **Smoke before spend.** A 10-minute smoke config precedes every long run.
3. **Checkpoint every 30 min**, resume automatically, prune to last-2 + best.
4. **Adapters live in a Kaggle Dataset**, not `/kaggle/working`.
5. **Log every run** (config hash, git SHA, metrics) to one CSV in the repo. The
   submission results table is generated from it, never retyped.
6. Two accounts to double quota is against Kaggle ToS — use **Colab free as overflow**
   if a run overruns.

---

## 5. Strategy notes

- **Declare captioning, ship grounding anyway.** Captioning metrics are far more
  reliably earned by a fine-tuned VLM; VRSBench grounding Acc@IoU0.5 is harsh and would
  drag the normalised aggregate down. Grounding still ships as an unscored demo tool,
  because a representative query asks for it.
- **Resolution domain gap is the top technical risk.** BigEarthNet is 120×120 at 10 m;
  the hidden ISRO set is Cartosat-2S sub-metre. Train on BigEarthNet for multisensor
  adaptation *and* VRSBench (DOTA/DIOR, ~0.3–1 m) for resolution robustness, with
  aggressive scale augmentation. Never train only on BigEarthNet.
- **fp16 instability** on T4 (no bf16) is a real failure mode. If loss goes NaN: lower
  LR, raise GradScaler init scale, keep the LM head in fp32.

---

## 6. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| BigEarthNet.txt schema differs from assumption | High | Adapter-style loader, one file to change |
| Quota exhausted mid-week | Medium | Weekly cap in plan + Colab overflow |
| fp16 divergence on T4 | Medium | fp32 head, LR sweep in smoke, loss-spike guard |
| 12 h cap kills a long run | High | Checkpoint/resume built in Stage 0 |
| Adapter underperforms baseline | Medium | Two shots then freeze; ship better of tuned/zero-shot |
| Kaggle dataset upload limits | Low | Shard prepared corpora, prune raw sources |

---

## 7. Parallel track

The agentic app does **not** wait for training. From W2 the app team builds against the
zero-shot base model behind the same interface the adapters will use, so swapping in
B1–B4 is a config change. Contracts, trace schema, geo I/O, compatibility checks, and
the UI are all independent of ML progress.
