# Session handoff — SatQuery AI (SIH26167)

Written 2026-09-12. Everything below was measured or read off the system, not
recalled. Where something is unverified it says so.

---

## 1. Where the project actually is

The programme has a **measured, defensible result**: the adapted model beats its
own unadapted base on identical splits, through identical code, with the
baseline re-measured in the same session as the adapted row.

| criterion | base | delivered (Stage B) | delta | *Stage A was* |
| --- | --- | --- | --- | --- |
| Single-image VQA | 0.2250 | **0.6675** | **+0.4425** | *+0.1825* |
| Change understanding | 0.0100 | **0.3000** | **+0.2900** | *+0.0350* |
| Optical–SAR joint | 0.0000 | **0.1600** | **+0.1600** | *+0.1750* |
| Grounding | 0.0000 | 0.0000 | 0.0000 | *0.0000* |
| Captioning | 0.0128 | 0.0006 | **−0.0123** | *−0.0126* |
| **Overall** | **0.0496** | **0.2256** | **+0.1761** | *+0.0760* |

`n=200, seed=1234, prompt 1.1.0, hf backend both rows.` Full provenance in
`results/2026-09-11-stage-b/`.

**The one claim to be careful with:** captioning is a *regression*, not a fix.
The deck says so and so should you.

---

## 2. Nothing is running

All compute is destroyed. Verified at handoff time.

| | state |
| --- | --- |
| vast.ai | 0 instances, **$5.03 credit** left of $10 |
| AWS us-east-1 | none |
| AWS ap-south-1 | none |
| IAM `satquery-box-stageb` | deleted (key, policy, user) |

**Cost so far:** vast.ai $4.97 for both training stages. AWS ~$23 for the
serving box, which ran 23 hours because the idle timer was disabled and never
re-armed — see §7.

---

## 3. Where the artefacts are

**The model** — `Siddu2004-2006/satquery-qwen3vl-2b` on Hugging Face, **private**.
4.26 GB, `Qwen3VLForConditionalGeneration`, bf16. This is the merged model: base
plus *both* adaptation stages folded in, loadable with a plain
`from_pretrained`. There is deliberately no separate Stage A / Stage B artefact —
Stage B resumed from Stage A's adapter, so it is one lineage.

**S3** — `s3://satquery-869987460914/` (ap-south-1):

| prefix | what |
| --- | --- |
| `adapters/stage-a/`, `adapters/stage-b/` | LoRA adapters, 197 MB each |
| `datasets/prepared/stage-b/train.jsonl` | the 64,000-record Stage B corpus |
| `datasets/prepared/stage-b-rehearsal/` | BigEarthNet at preamble-rate 0.85 |
| `datasets/prepared/train/` | Stage A corpus + 139,122 rendered images |
| `datasets/prepared-train.tar.gz` | the same images as one 5 GB object |
| `datasets/bigearthnet/` | the raw 155 GB LMDB |
| `results/` | both stages' CSVs, scores, quality diagnostics |

The merged weights are **not** in S3 — the results sync excludes `merged*/`
because a 4.3 GB artefact rebuildable from a 197 MB adapter is not worth storing.
Serving pulls from the Hub.

**Git** — branch `feat/ml-programme`, **85 commits ahead of `main`**, clean,
pushed. **509 tests passing.** `main` has none of this work; merging it is
outstanding and matters because `serve-model.sh` now defaults to that branch
name.

---

## 4. The permanent URL

`https://unconfessing-platiniferous-jeanelle.ngrok-free.dev` → the box's
`localhost:8000`, via an ngrok tunnel installed as a systemd service enabled at
boot. Credentials live in the parent-directory `.env`.

**The box it pointed at is terminated**, so the URL is dead until something is
redeployed. `serve-model.sh` does not yet install ngrok — that was done by hand
(`infra/`-adjacent script in the session scratch, not committed). **Folding the
ngrok setup into `serve-model.sh` is unfinished work.**

Why a tunnel at all: the security group admitted exactly one IP, and that IP
changed three times in one session, locking us out each time. ngrok connects
outbound, so port 8000 could be closed entirely.

---

## 5. What to do next, in order

**1. Retrain Stage B on the fixed corpus.** ~3 h, ~$0.65 of the $5.03. The
caption fix is written, tested and **never trained on**. This is the difference
between "captioning regressed, here is the fix, unvalidated" and a number.
Everything is ready: `infra/run-stage-b.sh` with `SKIP_BASELINE=1`.

**2. Redeploy for the demo.** `./infra/serve-model.sh --apply`. Every fix from
this session is in it. Budget an hour and do it *before* demo day — reviving a
`g5.xlarge` took six attempts on capacity alone.

**3. Merge `feat/ml-programme` to `main`.**

**4. Eyeball `docs/PRESENTATION_TECHNICAL.html`.** Structurally verified —
parses, all 14 nav links resolve — but never visually rendered; the browser
extension was down. Appendix C is new and table-heavy.

**5. Grounding is unsolved.** 0.000 before and after. The *format* problem is
fixed (unparsed boxes 33 → 1), so the model emits well-formed boxes on the right
grid in the wrong places. 10,000 referring records moved it not at all, so more
of the same data probably will not either. Either a different approach or an
honest "not solved" in the submission.

**6. CDVQA train/test tile overlap unresolved.** Shard 0 of each is disjoint,
but that is 3 images apiece against 122k questions drawn from a few thousand
SECOND tiles. Needs the full 82 GB pull for the guard to settle it. Change
understanding is already +0.2900 without it.

**7. Land-cover CNN untrained.** ~3 GPU-hours. It is the reason
`landcover_classes` is the one evidence key that never populates.

---

## 6. Access and credentials

Everything is in `C:/Users/sudik/OneDrive/Desktop/SIH26167/.env` — the **parent**
of the repo, so it cannot be committed.

`VAST_API_KEY`, `HF_TOKEN`, `NGROK_AUTHTOKEN`, `NGROK_DOMAIN`, `NGROK_DOMAIN_ID`.

**Parsing trap:** the ngrok entries are written `NAME = value` with spaces around
the equals; the older two are `NAME="value"`. A `startswith("NAME=")` check
returns empty for the ngrok ones — which is exactly how an empty authtoken
reached ngrok and produced an anonymous tunnel on a random URL. Split on `=` and
strip both sides.

**SSH key:** `~/.ssh/satquery-serve-use1.pem` (ed25519, us-east-1). Also
`~/.ssh/id_ed25519_vast` for vast.ai. AWS quota is **8 vCPUs of G/VT in
us-east-1 only** — every artefact is in ap-south-1, so serving crosses regions.

---

## 7. Traps this session paid for — do not re-pay

**`aws.exe` on Windows writes CRLF.** An OpenSSH private key with CRLF fails with
`error in libcrypto`, naming neither the file nor the cause. Convert to LF. And
`chmod` is a no-op on NTFS — use `icacls` or OpenSSH refuses the key as
world-readable.

**Git Bash rewrites `/dev/sda1`** into `C:/Program Files/Git/dev/sda1`. Pass
block-device mappings via a JSON file.

**Do not pin a subnet when launching GPU instances.** Naming one fails with
`InsufficientInstanceCapacity` in every AZ — each error suggesting the others,
which also fail — while the identical request with no subnet succeeds
immediately.

**Never send launch stderr to `/dev/null` in a retry loop.** A
`ParamValidation` error was reported as "no capacity" across thirty attempts.
Every failure looks like the same wrong diagnosis.

**The Deep Learning AMI ships Python 3.10**; `pyproject` needs ≥3.11 (`StrEnum`,
`datetime.UTC`). Install deadsnakes. This cost the same debugging on two
separate boxes because the fix was not carried across.

**vLLM needs `ninja`** for kernel warm-up or the engine dies at startup. This is
what defeated Stage A, where the workaround was falling back to the `hf`
backend. It is one `pip install`.

**`git clone` takes the default branch.** The first deploy served the new model
through the old application: health green, API answering, tool registry quietly
reporting 9 tools instead of 12.

**One `--root` override flattens every config's root.** The benchmark configs
carry different roots; overriding them all made every split unreadable, and the
contamination guard reported six SKIPPED splits and **exited 0**. That is fixed —
it exits non-zero when it cannot read its splits — but the lesson generalises.

**`wait` with no argument returns 0 even when a child failed.** Three
`ModuleNotFoundError` staging jobs went unnoticed this way.

**Verify a safety mechanism is *enabled*, not just written.** The idle timer was
disabled during debugging and never re-armed; the box then ran 23 hours at
$1/hr. I reported it as "armed" twice without checking the unit.

---

## 8. Design decisions worth not re-litigating

**Training on vast.ai, serving on AWS.** Training is bulk compute where price
dominates and a vanished machine costs one restart ($0.21/hr). Serving must be
up when someone points at it ($1.01/hr). Different requirements, different
infrastructure.

**The BigEarthNet rehearsal slice is not optional.** No benchmark train split
contains an optical–SAR pair, and no RGB benchmark image can support an evidence
preamble. Without the slice Stage B trains away what Stage A earned. It is why
cross-modal survived at 0.160.

**Question-only contamination overlap is advisory, not fatal.** All three corpora
are templated; BigEarthNet train and bench share 1,287 of 6,201 question strings
while sharing **zero** images and zero patches. Treating that as contamination
fails a clean corpus, and a guard that cries wolf gets switched off.

**The caption failure was verbosity, not truncation.** Re-running with the token
budget raised 128 → 320 gave CIDEr-D **0.000**, *worse* than the truncated 0.006.
One minute of GPU saved three hours spent on the wrong fix.
