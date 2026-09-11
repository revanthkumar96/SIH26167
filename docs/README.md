# SatQuery AI — documentation

An agentic vision-language assistant for multimodal remote-sensing analysis
through natural-language queries. Smart India Hackathon 2026, ISRO problem
statement **SIH26167**.

The system measures first and describes second: deterministic remote-sensing
tools compute quantities from pixels, and a vision-language model turns those
measurements into an answer. The controller decides which tools run, in what
order, and emits an auditable trace of everything it did.

---

## Start here

| Document | What it covers |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the system works — the six controller stages, evidence injection, contracts, serving |
| [ROADMAP.md](ROADMAP.md) | Status against the problem statement, phases, effort, cost, order of work |
| [PRESENTATION_TECHNICAL.md](PRESENTATION_TECHNICAL.md) | 10-minute presentation script with full technical depth, benchmarks, and metrics |

## By subject

| Document | What it covers |
| --- | --- |
| [TOOLS.md](TOOLS.md) | The nine registered tools, the registry contract, adaptive parameters, re-planning |
| [DATA.md](DATA.md) | Datasets, ingestion, cleaning, the evidence preamble, regional supervision |
| [FINETUNING.md](FINETUNING.md) | LoRA adaptation of Qwen3-VL — stages, hyperparameters, train/serve parity |
| [CNN.md](CNN.md) | The land-cover classifier and why the index tools need it |
| [AWS.md](AWS.md) | Hosting, training and serving on EC2; cost control; the demo-day runbook |
| [BENCHMARKING.md](BENCHMARKING.md) | The harness, metrics, prescribed splits, and how to report honestly |
| [SOURCES.md](SOURCES.md) | Every external dependency, with what was verified and what is still assumed |

---

## The short version

**What exists.** A working agentic application: uploads and compatibility
checking, a live Sentinel feed, single-image VQA / captioning / grounding,
bi-temporal change analysis, optical–SAR joint analysis, a tool registry with
enforced parameters, streaming execution traces, visual evidence, downloadable
reports, a benchmark harness over five prescribed splits, and 217 tests.

**What is missing.** Remote-sensing adaptation. The problem statement lists a
generic VLM with no fine-tuning under *Constraints and disqualifiers*, so this
is not an enhancement — it is the difference between a valid submission and an
invalid one. Everything in `FINETUNING.md`, `CNN.md` and `DATA.md` exists to
close it.

**What is settled.** BigEarthNet.txt is public and usable, its imagery is
available pre-encoded, and Qwen3-VL 2B trains under plain `transformers` +
`peft`. No blocking unknown remains.

**What is genuinely hard.** Domain distance. The adaptation data is Europe-only
at 10 m in 120×120 patches; the hidden evaluation set is Cartosat-2S and RISAT
over India at sub-metre resolution. No single dataset closes that, which is why
`DATA.md` describes three data programmes rather than one.

---

## Three things not to get wrong

**Run the baseline before training anything.** The base-versus-adapted comparison
on identical splits is the most important artefact this project produces. Without
a before column, adaptation is an unverifiable claim.

**Put the evidence preamble in the training data.** At inference the model is
handed measurements before its task prompt. A model adapted only on bare prompts
has never seen that, and will either ignore the measurements or grow more
confident in its own visual guess and contradict them. This is the same class of
bug as image-normalisation skew, in the prompt channel, and it would silently
waste the entire fine-tune.

**Report `n`, the seed, the prompt version and the model revision** with every
number. A score without them cannot be reproduced or defended, and the evaluation
is against a hidden reference we will not get to argue with.

---

## Conventions

Code and prose in these documents follow the repository's existing style:
comments and docs explain *why*, not *what*; deliberate deviations are recorded
as decisions rather than left as apparent omissions; and claims about external
sources are verified against the source and dated in `SOURCES.md`, with anything
still assumed marked **unverified**.
