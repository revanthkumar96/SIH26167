/* SatQuery AI — mission-control client (no build step) */

const state = {
  images: [],
  trace: null,
  socket: null,
  models: [],
  benchmarks: [],
  activeModel: null,
};

const PAGE_META = {
  mission: ["Mission control", "Query remote-sensing imagery in natural language"],
  models: ["Model catalog", "Pull weights and run multi-model bake-offs"],
  benchmarks: ["Benchmarks", "Evaluate against VRSBench, RSVQA, and CDVQA"],
  results: ["Leaderboard", "Headline metrics from runs/results.csv"],
  history: ["Session history", "Recent analysis and benchmark jobs"],
  registry: ["Tool registry", "Predefined specialist tools and permitted parameters"],
};

const EXAMPLES = [
  "Describe the land cover and major objects visible in this image.",
  "Highlight the water body referred to in the query.",
  "What changed between these two dates, and where did the change occur?",
  "Use the optical and SAR images together to identify built-up and water-covered regions.",
  "Has the built-up area increased, decreased, or remained unchanged?",
];

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const MB = 1024 * 1024;
const bytes = (n) =>
  n >= 1024 * MB ? `${(n / (1024 * MB)).toFixed(1)} GB` : `${Math.round(n / MB)} MB`;

async function errorDetail(res, fallback = "request failed") {
  try {
    const data = await res.json();
    return data.detail || data.message || fallback;
  } catch {
    return fallback;
  }
}

/* ── Navigation ────────────────────────────────────────────────────── */

document.querySelectorAll(".nav-item").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((t) => t.classList.remove("is-active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("is-active"));
    tab.classList.add("is-active");
    $(tab.dataset.tab).classList.add("is-active");
    const [title, subtitle] = PAGE_META[tab.dataset.tab] || ["SatQuery AI", ""];
    $("page-title").textContent = title;
    $("page-subtitle").textContent = subtitle;
    if (tab.dataset.tab === "results") loadResults();
    if (tab.dataset.tab === "history") loadHistory();
  });
});

/* ── Health & status ───────────────────────────────────────────────── */

async function loadHealth() {
  const badge = $("status");
  try {
    const data = await (await fetch("/api/health")).json();
    const base = `${data.settings.backend} · ${data.tools} tools`;
    renderStatus(badge, base, data.model);
    $("active-model").textContent = `${data.settings.backend} / ${data.settings.model}`;
    state.activeModel = data.settings;
    if (data.model && ["checking", "downloading"].includes(data.model.state)) {
      pollModel(badge, base);
    }
  } catch {
    badge.textContent = "offline";
    badge.className = "status-pill is-err";
  }
}

function renderStatus(badge, base, model) {
  if (!model || ["skipped", "ready", "idle"].includes(model.state)) {
    badge.textContent = base;
    badge.className = "status-pill is-ok";
    return;
  }
  if (model.state === "error") {
    badge.textContent = model.detail.slice(0, 40);
    badge.className = "status-pill is-err";
    return;
  }
  const pct = model.percent == null ? "" : ` ${model.percent}%`;
  badge.textContent =
    model.state === "checking" ? `checking ${model.model}…` : `downloading${pct}`;
  badge.className = "status-pill is-busy";
}

function pollModel(badge, base) {
  const timer = setInterval(async () => {
    try {
      const model = await (await fetch("/api/model")).json();
      renderStatus(badge, base, model);
      if (!["checking", "downloading"].includes(model.state)) {
        clearInterval(timer);
        loadModels();
      }
    } catch {
      clearInterval(timer);
    }
  }, 1500);
}

/* ── Models catalog ────────────────────────────────────────────────── */

async function loadModels() {
  const host = $("model-catalog");
  try {
    const data = await (await fetch("/api/models")).json();
    state.models = data.catalog;
    host.innerHTML = "";
    data.catalog.forEach((model) => host.append(modelCard(model, data.active)));
    renderBenchModelPicker(data.catalog);
  } catch {
    host.innerHTML = '<p class="empty">Could not load model catalog.</p>';
  }
}

function modelCard(model, active) {
  const card = el("div", `model-card${model.ready ? " is-ready" : ""}`);
  if (active && active.model === model.model && active.backend === model.backend) {
    card.classList.add("is-active");
  }

  const head = el("div", "model-card-head");
  const titles = el("div");
  titles.append(el("h3", null, model.label));
  titles.append(el("div", "model-id", `${model.backend} · ${model.model}`));
  head.append(titles);
  head.append(
    el(
      "span",
      `badge ${model.ready ? "badge-ok" : "badge-muted"}`,
      model.ready ? "ready" : model.backend === "echo" ? "built-in" : "not pulled"
    )
  );
  card.append(head);

  card.append(el("p", "model-desc", model.description));

  const meta = el("div", "model-meta");
  meta.append(el("span", "tag", model.params));
  meta.append(el("span", "tag", `VRAM ${model.vram}`));
  (model.tags || []).forEach((t) => meta.append(el("span", "tag", t)));
  if (model.size_bytes > 0) meta.append(el("span", "tag", bytes(model.size_bytes)));
  card.append(meta);

  const actions = el("div", "model-actions");
  if (model.backend !== "echo" && !model.ready) {
    const pull = el("button", "btn btn-ghost btn-sm", "Pull weights");
    pull.addEventListener("click", (e) => {
      e.stopPropagation();
      pullModel(model, pull);
    });
    actions.append(pull);
  }
  const test = el("button", "btn btn-ghost btn-sm", "Test in bake-off");
  test.addEventListener("click", () => {
    document.querySelector('[data-tab="benchmarks"]').click();
    setTimeout(() => selectBenchModel(model), 100);
  });
  actions.append(test);
  card.append(actions);
  return card;
}

async function pullModel(model, button) {
  button.disabled = true;
  button.textContent = "Pulling…";
  try {
    const res = await fetch("/api/models/pull", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: model.model }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "pull failed");
    await watchJob(data.run_id, () => {
      button.textContent = "Ready";
      loadModels();
    });
  } catch (err) {
    button.textContent = "Failed";
    button.disabled = false;
    showWarnings([String(err.message || err)]);
  }
}

function renderBenchModelPicker(catalog) {
  const host = $("bench-model-picker");
  if (!host) return;
  host.innerHTML = "";
  const label = el("div", "label-row", "Models to evaluate");
  host.before(label);
  catalog.forEach((model) => {
    const pick = el("label", "model-pick");
    const box = el("input");
    box.type = "checkbox";
    box.value = JSON.stringify({ backend: model.backend, model: model.model });
    box.checked = model.ready;
    box.dataset.modelId = model.id;
    pick.append(box, el("span", null, model.label));
    if (!model.ready && model.backend !== "echo") pick.style.opacity = "0.5";
    host.append(pick);
  });
}

function selectBenchModel(model) {
  document.querySelectorAll("#bench-model-picker input").forEach((input) => {
    const spec = JSON.parse(input.value);
    input.checked = spec.model === model.model && spec.backend === model.backend;
  });
}

function selectedBenchModels() {
  return Array.from(document.querySelectorAll("#bench-model-picker input:checked")).map(
    (input) => JSON.parse(input.value)
  );
}

/* ── Bake-off (Models tab) ─────────────────────────────────────────── */

async function loadBakeoffBenchmarks() {
  const host = $("bakeoff-bench-list");
  if (!host) return;
  try {
    const { benchmarks } = await (await fetch("/api/benchmarks")).json();
    host.innerHTML = "";
    benchmarks.forEach((bench) => {
      const row = el("label", "check-item");
      const box = el("input");
      box.type = "checkbox";
      box.value = bench.config;
      box.checked = Boolean(bench.data_present);
      row.append(
        box,
        el("span", "bench-name", bench.name),
        el("span", "bench-task", bench.task || ""),
        el(
          "span",
          `bench-avail ${bench.data_present ? "is-yes" : "is-no"}`,
          bench.data_present ? "ready" : "needs data"
        )
      );
      host.append(row);
    });
  } catch {
    host.innerHTML = '<p class="empty">Could not load benchmarks.</p>';
  }
}

$("bakeoff-run")?.addEventListener("click", async () => {
  const configs = Array.from(
    document.querySelectorAll("#bakeoff-bench-list input:checked")
  ).map((i) => i.value);
  const log = $("bakeoff-progress");
  log.hidden = false;
  log.textContent = "Starting bake-off…\n";

  const readyModels = state.models
    .filter((m) => m.ready)
    .map((m) => ({ backend: m.backend, model: m.model }));

  if (!configs.length) {
    log.textContent = "Select at least one benchmark.";
    return;
  }
  if (!readyModels.length) {
    log.textContent = "No ready models. Pull weights first or use echo baseline.";
    return;
  }

  await runBenchmarkJob({
    configs,
    limit: Number($("bakeoff-limit").value) || 200,
    models: readyModels,
    onEvent: (payload) => appendLog(log, payload),
    onComplete: () => loadResults(),
  });
});

/* ── Samples & upload ──────────────────────────────────────────────── */

async function loadSamples() {
  try {
    const { samples } = await (await fetch("/api/samples")).json();
    if (!samples.length) return;
    const host = $("sample-cards");
    host.innerHTML = "";
    samples.forEach((sample) => {
      const card = el("button", "sample-card");
      card.type = "button";
      card.append(
        el("span", "sc-config", sample.config),
        el("span", "sc-title", sample.title),
        el("span", "sc-sub", sample.subtitle)
      );
      card.addEventListener("click", () => loadSample(sample, card));
      host.append(card);
    });
    $("sample-strip").hidden = false;
  } catch {
    /* optional */
  }
}

async function loadSample(sample, card) {
  const label = card.querySelector(".sc-title").textContent;
  card.querySelector(".sc-title").textContent = "loading…";
  try {
    const res = await fetch(`/api/samples/${sample.id}/load`, { method: "POST" });
    if (!res.ok) throw new Error(await errorDetail(res, "could not load sample"));
    const data = await res.json();
    state.images = data.images;
    renderScenes();
    if (data.query) $("query").value = data.query;
    document.querySelector('[data-tab="mission"]').click();
  } catch (err) {
    showWarnings([String(err.message || err)]);
  } finally {
    card.querySelector(".sc-title").textContent = label;
  }
}

const dropzone = $("dropzone");
["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("is-over");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-over");
  })
);
dropzone.addEventListener("drop", (e) => upload(e.dataTransfer.files));
$("file-input").addEventListener("change", (e) => upload(e.target.files));

async function upload(fileList) {
  const files = Array.from(fileList).slice(0, 2);
  if (!files.length) return;
  const body = new FormData();
  files.forEach((f) => body.append("files", f));
  $("scenes").innerHTML = '<p class="empty">Reading imagery…</p>';
  try {
    const res = await fetch("/api/upload", { method: "POST", body });
    if (!res.ok) throw new Error(await errorDetail(res, "upload failed"));
    const data = await res.json();
    state.images = data.images;
    renderScenes();
  } catch (err) {
    $("scenes").innerHTML = "";
    showWarnings([String(err.message || err)]);
  }
}

function renderScenes() {
  const host = $("scenes");
  host.innerHTML = "";
  state.images.forEach((image) => {
    const card = el("div", "scene");
    const img = el("img");
    img.src = image.preview;
    img.alt = image.filename;
    img.loading = "lazy";
    const body = el("div", "scene-body");
    body.append(el("div", "scene-name", image.filename));
    const facts = el("div", "facts");
    const i = image.info;
    [i.size[0] + "×" + i.size[1], `${i.bands}b`, i.dtype, i.crs || i.format]
      .filter(Boolean)
      .forEach((f) => facts.append(el("span", "fact", f)));
    body.append(facts);
    card.append(img, body);
    host.append(card);
  });
  const banner = $("config-banner");
  if (state.images.length === 2) {
    banner.innerHTML = "";
    banner.append(
      document.createTextNode("Two scenes — "),
      el("strong", null, "bi-temporal"),
      document.createTextNode(" or "),
      el("strong", null, "cross-modal"),
      document.createTextNode(" inferred at run time.")
    );
    banner.hidden = false;
  } else {
    banner.hidden = true;
  }
  $("run").disabled = state.images.length === 0;
}

/* ── Query ─────────────────────────────────────────────────────────── */

const chips = $("examples");
EXAMPLES.forEach((text) => {
  const chip = el("button", "chip", text.length > 50 ? text.slice(0, 48) + "…" : text);
  chip.type = "button";
  chip.title = text;
  chip.addEventListener("click", () => {
    $("query").value = text;
    $("query").focus();
  });
  chips.append(chip);
});

$("run").addEventListener("click", runQuery);
$("query").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !$("run").disabled) runQuery();
});

async function runQuery() {
  const button = $("run");
  button.disabled = true;
  button.classList.add("is-busy");
  button.querySelector(".btn-label").textContent = "Running";
  resetResults();
  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: $("query").value,
        image_ids: state.images.map((i) => i.id),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "query failed");
    openStream(data.run_id);
  } catch (err) {
    showWarnings([String(err.message || err)]);
    finishRun();
  }
}

function finishRun() {
  const button = $("run");
  button.disabled = state.images.length === 0;
  button.classList.remove("is-busy");
  button.querySelector(".btn-label").textContent = "Run analysis";
  loadHistory();
}

function resetResults() {
  $("placeholder").hidden = true;
  $("trace").innerHTML = "";
  $("trace-block").hidden = false;
  $("evidence-block").hidden = true;
  $("checks-block").hidden = true;
  $("answer-card").hidden = true;
  $("warnings").hidden = true;
  $("download-report").hidden = true;
  state.trace = null;
}

function openStream(runId) {
  if (state.socket) state.socket.close();
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/ws/runs/${runId}`);
  state.socket = socket;
  socket.onmessage = (event) => handleEvent(JSON.parse(event.data));
  socket.onerror = () => showWarnings(["Lost connection to run stream"]);
  socket.onclose = finishRun;
}

const HIDDEN_OUTPUTS = new Set(["answer", "grounded_in_evidence", "bands_used"]);

function handleEvent(event) {
  if (event.type === "step") appendStep(event.step);
  else if (event.type === "complete") renderTrace(event.trace);
  else if (event.type === "error") showWarnings([event.message]);
}

function appendStep(step) {
  const item = el("li");
  const head = el("div", "step-head");
  head.append(el("span", "step-tool", step.tool), el("span", "step-meta", `v${step.version}`));
  if (step.adapter) head.append(el("span", "step-meta", step.adapter));
  if (step.confidence != null) head.append(el("span", "step-meta", `${step.confidence}`));
  head.append(el("span", "step-time", `${step.duration_ms} ms`));
  item.append(head);
  const entries = Object.entries(step.outputs).filter(([k]) => !HIDDEN_OUTPUTS.has(k));
  if (entries.length) {
    const list = el("dl", "step-outputs");
    entries.forEach(([key, value]) => {
      list.append(
        el("dt", null, key),
        el("dd", null, typeof value === "object" ? JSON.stringify(value) : String(value))
      );
    });
    item.append(list);
  }
  $("trace").append(item);
}

function renderTrace(trace) {
  state.trace = trace;
  $("routed-task").textContent = trace.routed_task.replace(/_/g, " ");
  $("answer").textContent = trace.answer || "(no answer)";
  const conf = trace.confidence;
  $("confidence-value").textContent = conf == null ? "" : `${(conf * 100).toFixed(0)}%`;
  $("confidence-bar").style.width = conf == null ? "0%" : `${conf * 100}%`;
  $("routing-rule").textContent = `routed by ${trace.routing_rule}`;
  $("answer-card").hidden = false;
  renderChecks(trace.input_check);
  renderEvidence(trace.evidence);
  showWarnings(trace.input_check.warnings || []);
  $("download-report").hidden = false;
}

function renderChecks(check) {
  const host = $("checks");
  host.innerHTML = "";
  const row = (key, node) => {
    const line = el("div", "check-row");
    const val = el("div", "check-val");
    val.append(node);
    line.append(el("span", "check-key", key), val);
    host.append(line);
  };
  row("configuration", el("span", null, check.config.replace(/_/g, " ")));
  row(
    "co-registered",
    el("span", check.coregistered ? "yes" : "no", check.coregistered ? "yes" : "not confirmed")
  );
  const passed = el("span", "check-val");
  (check.checks_passed || []).forEach((c) => passed.append(el("span", "tag is-pass", c)));
  row("passed", passed);
  (check.images || []).forEach((image, index) => {
    const facts = el("span", "check-val");
    [image.role, image.modality, image.crs]
      .filter(Boolean)
      .forEach((f) => facts.append(el("span", "tag", f)));
    row(`image ${index + 1}`, facts);
  });
  $("checks-block").hidden = false;
}

function renderEvidence(evidence) {
  const host = $("evidence");
  host.innerHTML = "";
  if (!evidence?.length) {
    $("evidence-block").hidden = true;
    return;
  }
  evidence.forEach((item) => {
    const figure = el("figure");
    if (item.type === "mask" && item.uri) {
      const img = el("img");
      img.src = `/${item.uri}`;
      img.alt = item.label || "mask";
      figure.append(img);
    } else if (item.type === "bbox" && item.bbox) {
      const source = state.images[0];
      const wrap = el("div", "bbox-wrap");
      const img = el("img");
      img.src = source ? source.preview : "";
      const [x1, y1, x2, y2] = item.bbox;
      const box = el("div", "box");
      box.style.left = `${x1 * 100}%`;
      box.style.top = `${y1 * 100}%`;
      box.style.width = `${(x2 - x1) * 100}%`;
      box.style.height = `${(y2 - y1) * 100}%`;
      wrap.append(img, box);
      figure.append(wrap);
    }
    figure.append(el("figcaption", null, item.label || item.type));
    host.append(figure);
  });
  $("evidence-block").hidden = false;
}

function showWarnings(messages) {
  const host = $("warnings");
  if (!messages?.length) {
    host.hidden = true;
    return;
  }
  host.innerHTML = "";
  host.append(el("strong", null, "Warning"));
  const list = el("ul");
  messages.forEach((m) => list.append(el("li", null, m)));
  host.append(list);
  host.hidden = false;
}

$("download-report").addEventListener("click", () => {
  if (!state.trace) return;
  const blob = new Blob([JSON.stringify(state.trace, null, 2)], { type: "application/json" });
  const link = el("a");
  link.href = URL.createObjectURL(blob);
  link.download = `satquery-${state.trace.run_id}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});

/* ── Datasets ──────────────────────────────────────────────────────── */

async function loadDatasets() {
  const host = $("dataset-cards");
  try {
    const { datasets } = await (await fetch("/api/datasets")).json();
    host.innerHTML = "";
    datasets.forEach((ds) => host.append(datasetCard(ds)));
  } catch {
    host.innerHTML = '<p class="empty">Could not load datasets.</p>';
  }
}

function datasetCard(ds) {
  const card = el("div", "dataset");
  const head = el("div", "ds-head");
  head.append(el("span", "ds-title", ds.title));
  head.append(
    el("span", `ds-state ${ds.ready ? "is-ready" : "is-missing"}`, ds.ready ? "on disk" : "missing")
  );
  card.append(head);
  const meta = el("div", "ds-meta");
  meta.append(el("div", "ds-source", ds.provenance));
  const link = el("a", null, "source");
  link.href = ds.homepage;
  link.target = "_blank";
  link.rel = "noopener";
  meta.append(link);
  card.append(meta);
  card.append(el("div", "ds-note", `${ds.download_mb} MB`));
  const actions = el("div", "ds-actions");
  const button = el("button", "btn btn-ghost btn-sm", ds.ready ? "Re-fetch" : "Download");
  const bar = el("div", "ds-progress");
  const fill = el("span");
  bar.append(fill);
  const status = el("span", "ds-status", "");
  actions.append(button, bar, status);
  card.append(actions);
  button.addEventListener("click", () => pullDataset(ds, button, fill, status));
  return card;
}

async function pullDataset(ds, button, fill, status) {
  button.disabled = true;
  status.textContent = "starting…";
  try {
    const res = await fetch("/api/datasets/pull", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: ds.name, with_images: false, shards: 2 }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "download failed");
    await watchJob(data.run_id, (payload) => {
      const p = payload.progress;
      if (p) {
        fill.style.width = `${p.percent || 0}%`;
        status.textContent = p.state === "downloading" ? `${p.percent || 0}%` : p.detail || p.state;
      }
    });
    status.textContent = "ready";
    button.disabled = false;
    button.textContent = "Re-fetch";
    loadDatasets();
    loadBenchmarks();
    loadBakeoffBenchmarks();
  } catch (err) {
    status.textContent = String(err.message || err);
    button.disabled = false;
  }
}

/* ── Benchmarks ────────────────────────────────────────────────────── */

async function loadBenchmarks() {
  const host = $("bench-list");
  try {
    const { benchmarks } = await (await fetch("/api/benchmarks")).json();
    state.benchmarks = benchmarks;
    host.innerHTML = "";
    if (!benchmarks.length) {
      host.innerHTML = '<p class="empty">No benchmark configs found.</p>';
      return;
    }
    benchmarks.forEach((bench) => {
      const row = el("label", "check-item");
      const box = el("input");
      box.type = "checkbox";
      box.value = bench.config;
      box.checked = Boolean(bench.data_present);
      row.append(
        box,
        el("span", "bench-name", bench.name),
        el("span", "bench-task", bench.task || ""),
        el(
          "span",
          `bench-avail ${bench.data_present ? "is-yes" : "is-no"}`,
          bench.data_present ? "ready" : "needs data"
        )
      );
      host.append(row);
    });
  } catch {
    host.innerHTML = '<p class="empty">Could not load benchmarks.</p>';
  }
}

$("bench-run").addEventListener("click", async () => {
  const configs = Array.from(document.querySelectorAll("#bench-list input:checked")).map(
    (i) => i.value
  );
  const models = selectedBenchModels();
  const live = $("bench-live");
  const output = $("bench-output");
  const cards = $("bench-results-cards");
  const bar = $("bench-progress-bar").firstElementChild;

  if (!configs.length) {
    output.hidden = false;
    output.textContent = "Select at least one benchmark.";
    return;
  }
  if (!models.length) {
    output.hidden = false;
    output.textContent = "Select at least one model.";
    return;
  }

  live.hidden = false;
  output.hidden = true;
  cards.innerHTML = "";
  bar.style.width = "0%";

  await runBenchmarkJob({
    configs,
    limit: Number($("bench-limit").value) || 200,
    models,
    onEvent: (payload) => {
      if (payload.type === "benchmark_result") {
        const headline = payload.metrics.oa ?? payload.metrics["acc@0.5"] ?? payload.metrics.cider_d;
        const metric = payload.metrics.oa != null ? "OA" : payload.metrics["acc@0.5"] != null ? "Acc@0.5" : "CIDEr";
        const card = el("div", "metric-card");
        card.append(
          el("div", "mc-name", `${payload.name} · ${payload.model}`),
          el("div", "mc-value", headline != null ? Number(headline).toFixed(3) : "—"),
          el("div", "mc-sub", `${metric} · ${payload.num_samples} samples · ${payload.duration_s}s`)
        );
        cards.append(card);
      }
      if (payload.type === "model_start") {
        bar.style.width = "10%";
      }
      if (payload.type === "benchmark_start") {
        bar.style.width = "40%";
      }
      if (payload.type === "complete") {
        bar.style.width = "100%";
        loadResults();
      }
    },
  });
});

async function runBenchmarkJob({ configs, limit, models, onEvent, onComplete }) {
  try {
    const res = await fetch("/api/benchmarks/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ configs, limit, models }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "benchmark failed");
    await watchJob(data.run_id, onEvent);
    if (onComplete) onComplete();
  } catch (err) {
    if (onEvent) onEvent({ type: "error", message: String(err.message || err) });
  }
}

function appendLog(host, payload) {
  if (payload.type === "model_start") {
    host.textContent += `\n▸ ${payload.model} (${payload.backend})\n`;
  } else if (payload.type === "benchmark_result") {
    const oa = payload.metrics.oa ?? payload.metrics["acc@0.5"] ?? payload.metrics.cider_d;
    host.textContent += `  ${payload.name}: ${oa != null ? Number(oa).toFixed(4) : "—"}\n`;
  } else if (payload.type === "complete") {
    host.textContent += `\n✓ Complete\n`;
  } else if (payload.type === "error") {
    host.textContent += `\n✗ ${payload.message}\n`;
  }
  host.scrollTop = host.scrollHeight;
}

function watchJob(runId, onEvent) {
  return new Promise((resolve) => {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${location.host}/ws/runs/${runId}`);
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (onEvent) onEvent(payload);
      if (payload.type === "complete" || payload.type === "error") {
        socket.close();
        resolve(payload);
      }
    };
    socket.onerror = () => resolve({ type: "error" });
  });
}

/* ── Results leaderboard ───────────────────────────────────────────── */

async function loadResults() {
  const wrap = $("results-table-wrap");
  const runsHost = $("results-runs");
  try {
    const data = await (await fetch("/api/results")).json();
    if (!data.models?.length) {
      wrap.innerHTML = '<p class="empty">No results yet. Run a benchmark or bake-off.</p>';
      runsHost.innerHTML = "";
      return;
    }

    const benchmarks = data.benchmarks;
    const best = {};
    benchmarks.forEach((bench) => {
      let max = -1;
      data.models.forEach((m) => {
        const v = m.scores[bench];
        if (v != null && v > max) max = v;
      });
      best[bench] = max;
    });

    const table = el("table", "data-table");
    const thead = el("thead");
    const headRow = el("tr");
    headRow.append(el("th", null, "Model"));
    benchmarks.forEach((b) => headRow.append(el("th", null, b)));
    thead.append(headRow);
    table.append(thead);

    const tbody = el("tbody");
    data.models.forEach((m) => {
      const row = el("tr");
      row.append(el("td", null, `${m.model} (${m.backend})`));
      benchmarks.forEach((bench) => {
        const val = m.scores[bench];
        const cell = el("td");
        const span = el(
          "span",
          `score${val != null && val === best[bench] ? " score-best" : ""}`,
          val != null ? val.toFixed(4) : "—"
        );
        cell.append(span);
        row.append(cell);
      });
      tbody.append(row);
    });
    table.append(tbody);
    wrap.innerHTML = "";
    wrap.append(table);

    runsHost.innerHTML = "";
    (data.runs || []).slice(0, 12).forEach((run) => {
      const entry = el("div", "run-entry");
      entry.append(el("time", null, run.timestamp));
      entry.append(el("span", "run-model", `${run.model} · ${run.backend}`));
      const scores = el("div", "run-scores");
      Object.entries(run.benchmarks || {}).forEach(([name, info]) => {
        scores.append(
          el("span", "run-score", `${name}: ${info.metric}=${info.value.toFixed(4)}`)
        );
      });
      entry.append(scores);
      runsHost.append(entry);
    });
  } catch {
    wrap.innerHTML = '<p class="empty">Could not load results.</p>';
  }
}

$("results-refresh").addEventListener("click", loadResults);

/* ── History ───────────────────────────────────────────────────────── */

async function loadHistory() {
  const host = $("history-list");
  try {
    const { runs } = await (await fetch("/api/runs")).json();
    host.innerHTML = "";
    if (!runs.length) {
      host.innerHTML = '<p class="empty">No runs yet.</p>';
      return;
    }
    runs.forEach((run) => {
      const item = el("div", "history-item");
      const kind = el(
        "span",
        `history-kind ${run.kind === "query" ? "is-query" : "is-benchmark"}`,
        run.kind
      );
      let summary = run.query || run.configs?.join(", ") || run.dataset || run.id;
      if (summary.length > 80) summary = summary.slice(0, 78) + "…";
      item.append(
        kind,
        el("span", "history-summary", summary),
        el("span", "history-status", run.status)
      );
      if (run.kind === "query") {
        item.addEventListener("click", () => openRunDetail(run.id));
      }
      host.append(item);
    });
  } catch {
    host.innerHTML = '<p class="empty">Could not load history.</p>';
  }
}

async function openRunDetail(runId) {
  try {
    const detail = await (await fetch(`/api/runs/${runId}`)).json();
    if (detail.result?.routed_task) {
      document.querySelector('[data-tab="mission"]').click();
      renderTrace(detail.result);
      $("placeholder").hidden = true;
    }
  } catch {
    /* ignore */
  }
}

/* ── Registry ──────────────────────────────────────────────────────── */

async function loadRegistry() {
  const host = $("registry-list");
  try {
    const { tools } = await (await fetch("/api/tools")).json();
    host.innerHTML = "";
    tools.forEach((tool) => {
      const card = el("div", "tool");
      const title = el("h4", null, tool.name);
      title.append(el("span", "version", ` v${tool.version}`));
      card.append(title);
      const rows = el("dl", "tool-rows");
      const add = (key, value) => {
        rows.append(el("dt", null, key), el("dd", null, value));
      };
      add("accepts", tool.accepts.replace(/_/g, " "));
      add("tasks", tool.tasks.join(", "));
      const params = Object.entries(tool.allowed_params)
        .map(([k, v]) => `${k} ∈ ${JSON.stringify(v)}`)
        .join("\n");
      add("permitted", params || "none");
      if (tool.outputs.length) add("outputs", tool.outputs.join(", "));
      card.append(rows);
      host.append(card);
    });
  } catch {
    host.innerHTML = '<p class="empty">Could not load registry.</p>';
  }
}

/* ── Boot ──────────────────────────────────────────────────────────── */

loadHealth();
loadSamples();
loadModels();
loadDatasets();
loadBenchmarks();
loadBakeoffBenchmarks();
loadRegistry();
loadResults();
