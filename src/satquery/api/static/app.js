/* SatQuery AI — client.
 *
 * No build step and no framework: the demo has to start with one command on a
 * machine that may have no Node toolchain, and an offline venue is a real risk.
 *
 * Two things are deliberate. The execution trace arrives over a websocket step
 * by step rather than as one payload at the end, because watching the tools fire
 * in order is the point. And nothing keeps run history — a run is live state,
 * and benchmark numbers are what is worth keeping.
 */

const state = {
  images: [],
  trace: null,
  socket: null,
  area: null,
  feedScenes: [],
  feedSelection: [],
  models: new Set(),
};

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
const wsUrl = (path) =>
  `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`;

const EXAMPLES = [
  "Describe the land cover and major objects visible in this image.",
  "Highlight the water body referred to in the query.",
  "What changed between these two dates, and where did the change occur?",
  "Use the optical and SAR images together to identify built-up and water regions.",
  "Has the built-up area increased, decreased, or remained unchanged?",
];

/* ── tabs ──────────────────────────────────────────────────────────── */

function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("is-active", t.dataset.tab === name)
  );
  document.querySelectorAll(".panel").forEach((p) =>
    p.classList.toggle("is-active", p.id === name)
  );
  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.querySelectorAll(".tab").forEach((tab) =>
  tab.addEventListener("click", () => showTab(tab.dataset.tab))
);
document.querySelectorAll("[data-goto]").forEach((button) =>
  button.addEventListener("click", () => showTab(button.dataset.goto))
);

/* ── status ────────────────────────────────────────────────────────── */

async function loadHealth() {
  const badge = $("status");
  try {
    const data = await (await fetch("/api/health")).json();
    const base = `${data.settings.backend} · ${data.settings.model} · ${data.tools} tools`;
    renderStatus(badge, base, data.model);
    if (data.model && ["checking", "downloading"].includes(data.model.state)) {
      pollModel(badge, base);
    }
  } catch {
    badge.textContent = "backend unreachable";
    badge.className = "status is-err";
  }
}

function renderStatus(badge, base, model) {
  if (!model || ["skipped", "ready", "idle"].includes(model.state)) {
    badge.textContent = base;
    badge.className = "status is-ok";
    return;
  }
  if (model.state === "error") {
    badge.textContent = `model unavailable — ${model.detail}`;
    badge.className = "status is-err";
    return;
  }
  const pct = model.percent == null ? "" : ` ${model.percent}%`;
  const size = model.total_bytes
    ? ` (${bytes(model.downloaded_bytes)} / ${bytes(model.total_bytes)})`
    : "";
  badge.textContent =
    model.state === "checking"
      ? `checking for ${model.model}…`
      : `downloading model${pct}${size}`;
  badge.className = "status is-busy";
}

function pollModel(badge, base) {
  const timer = setInterval(async () => {
    try {
      const model = await (await fetch("/api/model")).json();
      renderStatus(badge, base, model);
      if (!["checking", "downloading"].includes(model.state)) clearInterval(timer);
    } catch {
      clearInterval(timer);
    }
  }, 1500);
}

/* ── bundled samples ───────────────────────────────────────────────── */

async function loadSamples() {
  try {
    const { samples } = await (await fetch("/api/samples")).json();
    const host = $("sample-cards");
    host.innerHTML = "";
    if (!samples.length) {
      host.innerHTML =
        '<p class="empty">No bundled scenes. Use the live feed or drop your own.</p>';
      return;
    }
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
  } catch {
    /* samples are a convenience; absence is not an error */
  }
}

async function loadSample(sample, card) {
  const title = card.querySelector(".sc-title");
  const original = title.textContent;
  title.textContent = "loading…";
  try {
    const res = await fetch(`/api/samples/${sample.id}/load`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "could not load sample");
    state.images = data.images;
    renderScenes();
    if (data.query) $("query").value = data.query;
  } catch (err) {
    showWarnings([String(err.message || err)]);
  } finally {
    title.textContent = original;
  }
}

/* ── upload ────────────────────────────────────────────────────────── */

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
  $("scenes").innerHTML = '<p class="empty">reading imagery…</p>';
  try {
    const res = await fetch("/api/upload", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "upload failed");
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

    const i = image.info;
    const facts = el("div", "facts");
    [
      `${i.size[0]}×${i.size[1]}`,
      `${i.bands} band${i.bands === 1 ? "" : "s"}`,
      i.dtype,
      i.crs || i.format,
      i.gsd_m ? `${i.gsd_m} m/px` : null,
    ]
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
      document.createTextNode("Two scenes loaded. The configuration — "),
      el("strong", null, "bi-temporal"),
      document.createTextNode(" or "),
      el("strong", null, "cross-modal"),
      document.createTextNode(" — is inferred from their modality when you run.")
    );
    banner.hidden = false;
  } else {
    banner.hidden = true;
  }
  $("run").disabled = state.images.length === 0;
}

/* ── query ─────────────────────────────────────────────────────────── */

EXAMPLES.forEach((text) => {
  const chip = el("button", "chip", text.length > 52 ? `${text.slice(0, 50)}…` : text);
  chip.type = "button";
  chip.title = text;
  chip.addEventListener("click", () => {
    $("query").value = text;
    $("query").focus();
  });
  $("examples").append(chip);
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
}

function resetResults() {
  $("result-stage").hidden = false;
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
  const socket = new WebSocket(wsUrl(`/ws/runs/${runId}`));
  state.socket = socket;
  socket.onmessage = (event) => handleEvent(JSON.parse(event.data));
  socket.onerror = () => showWarnings(["lost connection to the run stream"]);
  socket.onclose = finishRun;
}

function handleEvent(event) {
  if (event.type === "step") appendStep(event.step);
  else if (event.type === "complete") renderTrace(event.trace);
  else if (event.type === "error") showWarnings([event.message]);
}

const HIDDEN_OUTPUTS = new Set(["answer", "grounded_in_evidence", "bands_used"]);

function appendStep(step) {
  const item = el("li");
  const head = el("div", "step-head");
  head.append(el("span", "step-tool", step.tool), el("span", "step-meta", `v${step.version}`));
  if (step.adapter) head.append(el("span", "step-meta", `adapter ${step.adapter}`));
  if (step.confidence != null) head.append(el("span", "step-meta", `conf ${step.confidence}`));
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
  $("answer").textContent = trace.answer || "(no answer produced)";
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
    line.append(el("span", "check-key", key));
    const val = el("div", "check-val");
    val.append(node);
    line.append(val);
    host.append(line);
  };

  row("configuration", el("span", null, check.config.replace(/_/g, " ")));
  row(
    "co-registered",
    el("span", check.coregistered ? "yes" : "no", check.coregistered ? "yes" : "not confirmed")
  );

  const passed = el("span", "check-val");
  (check.checks_passed || []).forEach((c) => passed.append(el("span", "tag is-pass", c)));
  row("checks passed", passed);

  (check.images || []).forEach((image, index) => {
    const facts = el("span", "check-val");
    [image.role, image.modality, `${image.size[0]}×${image.size[1]}`, `${image.bands}b`, image.crs]
      .filter(Boolean)
      .forEach((f) => facts.append(el("span", "tag", f)));
    row(`image ${index + 1}`, facts);
  });
  $("checks-block").hidden = false;
}

function renderEvidence(evidence) {
  const host = $("evidence");
  host.innerHTML = "";
  if (!evidence || !evidence.length) {
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
      img.alt = item.label || "region";
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
  if (!messages || !messages.length) {
    host.hidden = true;
    return;
  }
  $("result-stage").hidden = false;
  host.innerHTML = "";
  host.append(el("strong", null, "Warnings"));
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
  link.download = `satquery-report-${state.trace.run_id}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});

/* ── live feed ─────────────────────────────────────────────────────── */

async function loadFeed() {
  try {
    const { collections, areas } = await (await fetch("/api/feed/collections")).json();

    const select = $("feed-collection");
    select.innerHTML = "";
    collections.forEach((c) => {
      const option = el("option", null, `${c.label} — ${c.modality}`);
      option.value = c.id;
      select.append(option);
    });

    const host = $("feed-areas");
    host.innerHTML = "";
    areas.forEach((area, index) => {
      const card = el("button", "area-card");
      card.type = "button";
      card.append(
        el("span", "ac-label", area.label),
        el("span", "ac-note", area.note),
        el("span", "ac-bbox", area.bbox.map((v) => v.toFixed(2)).join(", "))
      );
      card.addEventListener("click", () => {
        state.area = area;
        document
          .querySelectorAll(".area-card")
          .forEach((c) => c.classList.remove("is-active"));
        card.classList.add("is-active");
        searchFeed();
      });
      host.append(card);
      if (index === 0) {
        state.area = area;
        card.classList.add("is-active");
      }
    });
  } catch {
    $("feed-areas").innerHTML = '<p class="empty">Could not load the feed.</p>';
  }
}

$("feed-search").addEventListener("click", searchFeed);

async function searchFeed() {
  if (!state.area) return;
  const results = $("feed-results");
  const hint = $("feed-hint");
  results.innerHTML = '<p class="empty">searching the catalogue…</p>';
  hint.hidden = true;
  state.feedSelection = [];
  updateFeedSelection();

  const days = Number($("feed-days").value) || 60;
  const start = new Date(Date.now() - days * 86400000).toISOString().slice(0, 10);

  try {
    const res = await fetch("/api/feed/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        collection: $("feed-collection").value,
        bbox: state.area.bbox,
        start,
        max_cloud: Number($("feed-cloud").value),
        limit: 12,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "search failed");

    state.feedScenes = data.scenes;
    renderFeedScenes(data.scenes);
    if (data.hint) {
      hint.textContent = data.hint;
      hint.hidden = false;
    }
  } catch (err) {
    results.innerHTML = "";
    hint.textContent = String(err.message || err);
    hint.hidden = false;
  }
}

function renderFeedScenes(scenes) {
  const host = $("feed-results");
  host.innerHTML = "";
  if (!scenes.length) {
    host.innerHTML = '<p class="empty">No acquisitions matched.</p>';
    return;
  }

  scenes.forEach((scene) => {
    const card = el("button", "feed-scene");
    card.type = "button";
    card.append(el("span", "fs-order", ""));

    if (scene.preview) {
      const img = el("img");
      img.src = scene.preview;
      img.alt = scene.id;
      img.loading = "lazy";
      card.append(img);
    }

    const body = el("div", "fs-body");
    body.append(el("div", "fs-date", scene.datetime.slice(0, 10)));
    const meta = el("div", "fs-meta");
    [
      scene.cloud_cover != null ? `${scene.cloud_cover.toFixed(0)}% cloud` : null,
      scene.platform,
      scene.orbit_state,
      scene.instrument_mode,
    ]
      .filter(Boolean)
      .forEach((m) => meta.append(el("span", "tag", m)));
    body.append(meta);
    card.append(body);

    card.addEventListener("click", () => toggleFeedScene(scene, card));
    host.append(card);
  });
}

function toggleFeedScene(scene, card) {
  const index = state.feedSelection.findIndex((s) => s.id === scene.id);
  if (index >= 0) {
    state.feedSelection.splice(index, 1);
  } else {
    // Two is the ceiling: the analyse path accepts one scene or one pair.
    if (state.feedSelection.length === 2) state.feedSelection.shift();
    state.feedSelection.push(scene);
  }
  document.querySelectorAll(".feed-scene").forEach((node) => {
    node.classList.remove("is-selected");
    node.querySelector(".fs-order").textContent = "";
  });
  const cards = Array.from(document.querySelectorAll(".feed-scene"));
  state.feedSelection.forEach((selected, order) => {
    const position = state.feedScenes.findIndex((s) => s.id === selected.id);
    const node = cards[position];
    if (node) {
      node.classList.add("is-selected");
      node.querySelector(".fs-order").textContent = String(order + 1);
    }
  });
  void card;
  updateFeedSelection();
}

function updateFeedSelection() {
  const label = $("feed-selection");
  const count = state.feedSelection.length;
  if (!count) {
    label.textContent = "nothing selected";
  } else if (count === 1) {
    label.textContent = "1 scene — single-image analysis";
  } else {
    const modalities = new Set(
      state.feedSelection.map((s) =>
        s.collection.startsWith("sentinel-1") ? "sar" : "optical"
      )
    );
    label.textContent =
      modalities.size === 2
        ? "2 scenes — cross-modal pair"
        : "2 scenes — bi-temporal pair";
  }
  $("feed-load").disabled = count === 0;
}

$("feed-load").addEventListener("click", async () => {
  if (!state.feedSelection.length || !state.area) return;

  const button = $("feed-load");
  const progress = $("feed-progress");
  const fill = progress.querySelector(".ds-progress span");
  const status = progress.querySelector(".ds-status");
  button.disabled = true;
  progress.hidden = false;
  status.textContent = "starting…";

  try {
    const res = await fetch("/api/feed/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scenes: state.feedSelection.map((s) => ({ id: s.id, collection: s.collection })),
        bbox: state.area.bbox,
        size: 1024,
        compact_optical: state.feedSelection.length === 2,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "load failed");

    const socket = new WebSocket(wsUrl(`/ws/runs/${data.run_id}`));
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.progress) {
        fill.style.width = `${payload.progress.percent || 0}%`;
        status.textContent = payload.progress.detail || payload.progress.state;
      }
      if (payload.type === "complete") {
        fill.style.width = "100%";
        status.textContent = payload.result.detail;
        state.images = payload.result.images;
        renderScenes();
        button.disabled = false;
        showTab("analyse");
      } else if (payload.type === "error") {
        status.textContent = payload.message;
        button.disabled = false;
      }
    };
  } catch (err) {
    status.textContent = String(err.message || err);
    button.disabled = false;
  }
});

/* ── datasets ──────────────────────────────────────────────────────── */

async function loadDatasets() {
  const host = $("dataset-cards");
  try {
    const { datasets } = await (await fetch("/api/datasets")).json();
    host.innerHTML = "";
    datasets.forEach((ds) => host.append(datasetCard(ds)));
  } catch {
    host.innerHTML = '<p class="empty">Could not load dataset sources.</p>';
  }
}

function datasetCard(ds) {
  const card = el("div", "dataset");
  const head = el("div", "ds-head");
  head.append(el("span", "ds-title", ds.title));
  head.append(
    el(
      "span",
      `ds-state ${ds.ready ? "is-ready" : "is-missing"}`,
      ds.ready ? "on disk" : "not fetched"
    )
  );
  card.append(head);

  const meta = el("div", "ds-meta");
  meta.append(el("div", "ds-source", ds.provenance));
  const link = el("a", null, ds.homepage);
  link.href = ds.homepage;
  link.target = "_blank";
  link.rel = "noopener";
  meta.append(link);
  card.append(meta);

  const size = ds.shards
    ? `${ds.shard_size_mb} MB per shard · ${ds.shards} shards available`
    : `${ds.download_mb} MB` +
      (ds.optional_mb ? ` · +${(ds.optional_mb / 1024).toFixed(1)} GB imagery optional` : "");
  card.append(el("div", "ds-note", size));
  if (ds.note) card.append(el("div", "ds-note", ds.note));

  const actions = el("div", "ds-actions");
  const button = el("button", "btn-ghost", ds.ready ? "Re-fetch" : "Download");
  const bar = el("div", "ds-progress");
  const fill = el("span");
  bar.append(fill);
  const status = el("span", "ds-status", "");
  actions.append(button, bar, status);
  card.append(actions);

  button.addEventListener("click", async () => {
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

      const socket = new WebSocket(wsUrl(`/ws/runs/${data.run_id}`));
      socket.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        const p = payload.progress;
        if (p) {
          fill.style.width = `${p.percent || 0}%`;
          status.textContent =
            p.state === "downloading" ? `${p.percent || 0}% · ${p.current_file}` : p.detail;
        }
        if (payload.type === "complete") {
          fill.style.width = "100%";
          status.textContent = "ready";
          button.disabled = false;
          loadDatasets();
          loadBenchmarks();
        } else if (payload.type === "error") {
          status.textContent = payload.message;
          button.disabled = false;
        }
      };
    } catch (err) {
      status.textContent = String(err.message || err);
      button.disabled = false;
    }
  });
  return card;
}

/* ── models ────────────────────────────────────────────────────────── */

async function loadModels() {
  const host = $("model-cards");
  try {
    const payload = await (await fetch("/api/models")).json();
    const models = payload.models || payload.catalog || [];
    host.innerHTML = "";
    models.forEach((m, index) => host.append(modelCard(m, index === 0)));
  } catch {
    host.innerHTML = '<p class="empty">Could not load the model catalog.</p>';
  }
}

function modelCard(model, preselect) {
  const card = el("div", "model-card");
  const head = el("div", "mc-head");

  const box = el("input");
  box.type = "checkbox";
  box.value = model.id;
  if (preselect) {
    box.checked = true;
    state.models.add(model.id);
    card.classList.add("is-selected");
  }
  box.addEventListener("change", () => {
    if (box.checked) state.models.add(model.id);
    else state.models.delete(model.id);
    card.classList.toggle("is-selected", box.checked);
  });

  head.append(box, el("span", "mc-label", model.label), el("span", "mc-params", model.params));
  const ready = model.ready ?? model.present;
  head.append(
    el(
      "span",
      `mc-state ${ready ? "is-ready" : "is-missing"}`,
      ready ? "on disk" : `${model.size_gb ?? "?"} GB`
    )
  );
  card.append(head);

  card.append(el("div", "mc-desc", model.description));
  card.append(el("div", "mc-repo", model.model));

  const foot = el("div", "mc-foot");
  foot.append(el("span", "mc-licence", `licence: ${model.license || "unknown"}`));
  if (!ready && model.backend !== "echo") {
    const pull = el("button", "btn-ghost", "Download weights");
    const status = el("span", "ds-status", "");
    pull.addEventListener("click", async () => {
      pull.disabled = true;
      status.textContent = "starting…";
      try {
        const res = await fetch("/api/models/pull", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: model.id, model: model.model }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "download failed");
        status.textContent = "downloading — see the header for progress";
      } catch (err) {
        status.textContent = String(err.message || err);
        pull.disabled = false;
      }
    });
    foot.append(pull, status);
  }
  card.append(foot);
  return card;
}

/* ── benchmarks ────────────────────────────────────────────────────── */

async function loadBenchmarks() {
  const host = $("bench-list");
  try {
    const { benchmarks } = await (await fetch("/api/benchmarks")).json();
    host.innerHTML = "";
    if (!benchmarks.length) {
      host.innerHTML = '<p class="empty">No benchmark configs found.</p>';
      return;
    }
    benchmarks.forEach((bench) => {
      const row = el("div", "bench-row");
      const box = el("input");
      box.type = "checkbox";
      box.value = bench.config;
      box.checked = Boolean(bench.data_present);
      row.append(box, el("span", "bench-name", bench.name));
      if (bench.task) row.append(el("span", "bench-task", bench.task));
      row.append(
        el(
          "span",
          `bench-avail ${bench.data_present ? "is-yes" : "is-no"}`,
          bench.error ? "config error" : bench.data_present ? "data present" : "data missing"
        )
      );
      host.append(row);
    });
  } catch {
    host.innerHTML = '<p class="empty">Could not load benchmarks.</p>';
  }
}

$("bench-run").addEventListener("click", async () => {
  const configs = Array.from(
    document.querySelectorAll("#bench-list input:checked")
  ).map((i) => i.value);
  const models = Array.from(state.models);

  const stage = $("matrix-stage");
  const progress = $("matrix-progress");
  const fill = progress.querySelector(".ds-progress span");
  const status = progress.querySelector(".ds-status");
  stage.hidden = false;

  if (!models.length || !configs.length) {
    status.textContent = "Select at least one model and one benchmark.";
    return;
  }

  $("matrix-table").innerHTML = "";
  fill.style.width = "0%";
  status.textContent = "starting…";

  try {
    const res = await fetch("/api/benchmarks/matrix", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        models,
        configs,
        limit: Number($("bench-limit").value) || 200,
        seed: Number($("bench-seed").value) || 1234,
        reuse_cached: $("bench-reuse").checked,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "matrix run failed");

    const socket = new WebSocket(wsUrl(`/ws/runs/${data.run_id}`));
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      const p = payload.progress || (payload.result && payload.result);
      if (p && p.cells) {
        fill.style.width = `${p.percent || 0}%`;
        status.textContent =
          `${p.cells_done}/${p.cells_total} · ${p.detail || p.state}` +
          (p.current_model ? ` · ${p.current_model}` : "");
        renderMatrix(p.cells);
      }
      if (payload.type === "error") status.textContent = payload.message;
    };
  } catch (err) {
    status.textContent = String(err.message || err);
  }
});

function renderMatrix(cells) {
  const benchmarks = [];
  const models = [];
  cells.forEach((c) => {
    if (!benchmarks.includes(c.benchmark)) benchmarks.push(c.benchmark);
    if (!models.includes(c.model_id)) models.push(c.model_id);
  });

  const lookup = new Map(cells.map((c) => [`${c.model_id}|${c.benchmark}`, c]));

  // Highlight the leader per benchmark, so the comparison reads at a glance.
  const best = new Map();
  benchmarks.forEach((b) => {
    let top = null;
    models.forEach((m) => {
      const cell = lookup.get(`${m}|${b}`);
      if (cell && cell.headline != null && (top == null || cell.headline > top)) {
        top = cell.headline;
      }
    });
    best.set(b, top);
  });

  const table = el("table", "matrix");
  const thead = el("thead");
  const headRow = el("tr");
  headRow.append(el("th", null, "model"));
  benchmarks.forEach((b) => {
    const cell = cells.find((c) => c.benchmark === b);
    headRow.append(el("th", null, `${b} · ${cell ? cell.headline_metric : ""}`));
  });
  thead.append(headRow);
  table.append(thead);

  const tbody = el("tbody");
  models.forEach((m) => {
    const row = el("tr");
    row.append(el("th", null, m));
    benchmarks.forEach((b) => {
      const cell = lookup.get(`${m}|${b}`);
      if (!cell || cell.status === "pending") {
        row.append(el("td", "cell-pending", "—"));
        return;
      }
      if (cell.status === "error") {
        row.append(el("td", "cell-error", cell.error.slice(0, 90)));
        return;
      }
      if (cell.status === "running") {
        row.append(el("td", "cell-pending", "running…"));
        return;
      }
      const value = cell.headline;
      const node = el(
        "td",
        `num${value != null && value === best.get(b) ? " best" : ""}`,
        value == null ? "—" : value.toFixed(4)
      );
      node.title = `${cell.num_samples} samples · ${cell.duration_s}s · ${cell.status}`;
      row.append(node);
    });
    tbody.append(row);
  });
  table.append(tbody);

  const host = $("matrix-table");
  host.innerHTML = "";
  host.append(table);
  host.append(
    el(
      "p",
      "metric-note",
      "Bold = best per benchmark. Hover a cell for sample count and runtime. " +
        "vqa/change_vqa report overall accuracy; caption reports CIDEr-D; " +
        "grounding reports Acc@IoU0.5."
    )
  );

  renderBreakdown(cells);
}

function renderBreakdown(cells) {
  // Per-question-type accuracy is where OA and AA diverge, so it is worth its
  // own table rather than a tooltip.
  const scored = cells.filter(
    (c) => c.metrics && Object.keys(c.metrics).some((k) => k.startsWith("acc/"))
  );
  if (!scored.length) {
    $("matrix-detail").hidden = true;
    return;
  }

  const table = el("table", "matrix");
  const thead = el("thead");
  const headRow = el("tr");
  ["model", "benchmark", "OA", "AA"].forEach((h) => headRow.append(el("th", null, h)));

  const types = [];
  scored.forEach((c) =>
    Object.keys(c.metrics)
      .filter((k) => k.startsWith("acc/"))
      .forEach((k) => {
        const name = k.slice(4);
        if (!types.includes(name)) types.push(name);
      })
  );
  types.forEach((t) => headRow.append(el("th", null, t)));
  thead.append(headRow);
  table.append(thead);

  const tbody = el("tbody");
  scored.forEach((c) => {
    const row = el("tr");
    row.append(el("th", null, c.model_id), el("td", null, c.benchmark));
    ["oa", "aa"].forEach((key) => {
      const v = c.metrics[key];
      row.append(el("td", "num", v == null ? "—" : v.toFixed(4)));
    });
    types.forEach((t) => {
      const v = c.metrics[`acc/${t}`];
      row.append(el("td", "num", v == null ? "—" : v.toFixed(3)));
    });
    tbody.append(row);
  });
  table.append(tbody);

  const host = $("matrix-breakdown");
  host.innerHTML = "";
  host.append(table);
  host.append(
    el(
      "p",
      "metric-note",
      "AA is the unweighted mean over question types. A model that always answers " +
        "the majority class posts a respectable OA and a poor AA."
    )
  );
  $("matrix-detail").hidden = false;
}

/* ── registry ──────────────────────────────────────────────────────── */

async function loadRegistry() {
  const host = $("registry-groups");
  try {
    const { tools } = await (await fetch("/api/tools")).json();

    const legend = $("registry-legend");
    legend.innerHTML = "";
    const measurement = el("span");
    measurement.append(el("span", "swatch"), document.createTextNode(
      "measurement — deterministic, reproducible, becomes evidence"
    ));
    const model = el("span");
    model.append(el("span", "swatch model"), document.createTextNode(
      "model — learned, checked against the measurements"
    ));
    legend.append(measurement, model);

    const groups = new Map();
    tools.forEach((tool) => {
      if (!groups.has(tool.category)) groups.set(tool.category, []);
      groups.get(tool.category).push(tool);
    });

    host.innerHTML = "";
    Array.from(groups.entries()).forEach(([category, items]) => {
      const group = el("div", "tool-group");
      group.append(el("h3", null, `${category} · ${items.length}`));
      const grid = el("div", "tool-grid");
      items.forEach((tool) => grid.append(toolCard(tool)));
      group.append(grid);
      host.append(group);
    });
  } catch {
    host.innerHTML = '<p class="empty">Could not load the registry.</p>';
  }
}

function toolCard(tool) {
  const card = el("div", "tool");

  const head = el("div", "tool-head");
  head.append(el("span", `tool-kind${tool.kind === "model" ? " model" : ""}`));
  head.append(el("span", "tool-name", tool.name), el("span", "tool-version", `v${tool.version}`));
  card.append(head);

  card.append(el("div", "tool-summary", tool.summary));

  const badges = el("div", "tool-badges");
  [
    tool.accepts.replace(/_/g, " "),
    tool.cost,
    ...tool.tasks,
    tool.emits_evidence ? "emits evidence" : null,
  ]
    .filter(Boolean)
    .forEach((b) => badges.append(el("span", "tag", b)));
  card.append(badges);

  if (tool.requires) card.append(el("div", "pdoc", `requires: ${tool.requires}`));

  if (tool.params.length) {
    const grid = el("div", "param-table");
    tool.params.forEach((p) => {
      grid.append(
        el("span", "pname", p.name),
        el("span", "pconstraint", p.constraint),
        el("span", "pdoc", p.doc)
      );
    });
    card.append(grid);
  }

  if (tool.outputs.length) {
    card.append(el("div", "pdoc", `outputs: ${tool.outputs.join(", ")}`));
  }
  return card;
}

loadHealth();
loadSamples();
loadFeed();
loadDatasets();
loadModels();
loadBenchmarks();
loadRegistry();
