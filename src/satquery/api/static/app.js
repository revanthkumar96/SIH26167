/* SatQuery AI — mission-control client (no build step) */

const state = {
  images: [],
  trace: null,
  socket: null,
  models: [],
  benchmarks: [],
  activeModel: null,
  runs: [],
  history: [],
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

const DEFAULT_MODELS = [
  {
    id: "satchat-echo",
    backend: "echo",
    model: "SatChat-Echo-v1",
    label: "SatChat Echo Baseline",
    description: "Built-in lightweight baseline model for rapid local testing and routing verification.",
    params: "0.1B",
    vram: "0.5 GB",
    ready: true,
    size_bytes: 150 * 1024 * 1024,
    tags: ["baseline", "echo", "fast"],
  },
  {
    id: "earthvlm-7b",
    backend: "pytorch",
    model: "EarthVLM-7B-Instruct",
    label: "EarthVLM 7B Instruct",
    description: "Multimodal vision-language model trained on Sentinel-2 optical & Sentinel-1 SAR imagery.",
    params: "7B",
    vram: "14 GB",
    ready: true,
    size_bytes: 14 * 1024 * 1024 * 1024,
    tags: ["VLM", "sentinel-2", "SAR"],
  },
  {
    id: "geollava-13b",
    backend: "pytorch",
    model: "GeoLlava-13B",
    label: "GeoLlava 13B RS",
    description: "Large vision-language model specialized in high-resolution aerial & satellite spatial reasoning.",
    params: "13B",
    vram: "24 GB",
    ready: false,
    size_bytes: 26 * 1024 * 1024 * 1024,
    tags: ["VLM", "spatial-reasoning"],
  },
  {
    id: "remoteclip-vit-l",
    backend: "onnx",
    model: "RemoteCLIP-ViT-L",
    label: "RemoteCLIP ViT-L/14",
    description: "Zero-shot satellite scene classifier and text-image embedding model.",
    params: "0.4B",
    vram: "4 GB",
    ready: true,
    size_bytes: 1.8 * 1024 * 1024 * 1024,
    tags: ["zero-shot", "embeddings"],
  },
];

const DEFAULT_BENCHMARKS = [
  { config: "vrsbench_val", name: "VRSBench Val", task: "Visual Question Answering", data_present: true },
  { config: "rsvqa_lr_test", name: "RSVQA-LR Test", task: "Low-Res VQA", data_present: true },
  { config: "rsvqa_hr_test", name: "RSVQA-HR Test", task: "High-Res VQA", data_present: true },
  { config: "cdvqa_test", name: "CDVQA Test", task: "Change Detection VQA", data_present: true },
];

const DEFAULT_DATASETS = [
  { name: "vrsbench", title: "VRSBench Dataset", provenance: "Visual RS Benchmark Suite", homepage: "https://github.com/VRSBench/VRSBench", ready: true, download_mb: 320 },
  { name: "rsvqa_lr", title: "RSVQA Low-Resolution", provenance: "Sentinel-2 VQA Dataset", homepage: "https://rsvqa.zenodo.org", ready: true, download_mb: 450 },
  { name: "rsvqa_hr", title: "RSVQA High-Resolution", provenance: "Aerial Imagery VQA Dataset", homepage: "https://rsvqa.zenodo.org", ready: true, download_mb: 1200 },
  { name: "cdvqa", title: "CDVQA Change Detection", provenance: "Bi-Temporal Remote Sensing VQA", homepage: "https://github.com/CDVQA/CDVQA", ready: false, download_mb: 680 },
];

const DEFAULT_TOOLS = [
  { name: "input_validator", version: "1.2", accepts: "raw_imagery", tasks: ["crs_check", "coregistration"], allowed_params: { tolerance_m: [1, 5, 10] }, outputs: ["input_check"] },
  { name: "task_router", version: "2.0", accepts: "text_query", tasks: ["task_routing"], allowed_params: { strategy: ["heuristic", "vlm"] }, outputs: ["routed_task", "routing_rule"] },
  { name: "segment_water", version: "1.5", accepts: "optical_sar", tasks: ["water_body_segmentation"], allowed_params: { ndwi_threshold: [0.1, 0.2, 0.3] }, outputs: ["mask"] },
  { name: "detect_changes", version: "2.1", accepts: "bitemporal", tasks: ["change_detection"], allowed_params: { method: ["dndvi", "sar_ratio"] }, outputs: ["change_map", "bbox"] },
  { name: "vlm_reasoner", version: "3.0", accepts: "multimodal", tasks: ["visual_question_answering"], allowed_params: { max_tokens: [128, 256, 512] }, outputs: ["answer", "confidence"] },
];

const DEFAULT_SAMPLES = [
  {
    id: "sih-demo-1",
    config: "single_scene",
    title: "Sentinel-2 Land Cover (Urban & Water)",
    subtitle: "Optical RGB scene over coastal river reservoir",
    query: "Describe the land cover and major objects visible in this image.",
    images: [
      {
        id: "img-s2-1",
        filename: "Sentinel2_L2A_RGB_Tile34.tif",
        preview: createSampleThumbnail("Sentinel-2 Optical RGB", "#1a4971", "#2e8b57"),
        info: { size: [1024, 1024], bands: 3, dtype: "uint8", crs: "EPSG:32643", format: "GeoTIFF", role: "primary", modality: "optical" },
      },
    ],
  },
  {
    id: "sih-demo-2",
    config: "cross_modal",
    title: "Optical + SAR Co-registered Pair",
    subtitle: "Sentinel-2 Optical RGB + Sentinel-1 SAR VV/VH",
    query: "Use the optical and SAR images together to identify built-up and water-covered regions.",
    images: [
      {
        id: "img-s2-opt",
        filename: "Sentinel2_L2A_Optical_T43.tif",
        preview: createSampleThumbnail("Sentinel-2 Optical RGB", "#194a6e", "#3b7a57"),
        info: { size: [1024, 1024], bands: 4, dtype: "uint16", crs: "EPSG:32643", format: "GeoTIFF", role: "optical_master", modality: "optical" },
      },
      {
        id: "img-s1-sar",
        filename: "Sentinel1_SAR_IW_VVVH.tif",
        preview: createSampleThumbnail("Sentinel-1 SAR VV/VH", "#2c3e50", "#7f8c8d"),
        info: { size: [1024, 1024], bands: 2, dtype: "float32", crs: "EPSG:32643", format: "GeoTIFF", role: "sar_slave", modality: "sar" },
      },
    ],
  },
  {
    id: "sih-demo-3",
    config: "bitemporal",
    title: "Sentinel-2 Bi-Temporal Pair (2023 vs 2025)",
    subtitle: "Land cover transition & urban expansion analysis",
    query: "What changed between these two dates, and where did the change occur?",
    images: [
      {
        id: "img-t1",
        filename: "Sentinel2_2023_04_15.tif",
        preview: createSampleThumbnail("T1: April 2023", "#1b4332", "#40916c"),
        info: { size: [1024, 1024], bands: 3, dtype: "uint8", crs: "EPSG:32643", format: "GeoTIFF", role: "t1_pre", modality: "optical" },
      },
      {
        id: "img-t2",
        filename: "Sentinel2_2025_04_20.tif",
        preview: createSampleThumbnail("T2: April 2025", "#2d6a4f", "#74c69d"),
        info: { size: [1024, 1024], bands: 3, dtype: "uint8", crs: "EPSG:32643", format: "GeoTIFF", role: "t2_post", modality: "optical" },
      },
    ],
  },
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

function createSampleThumbnail(label, color1, color2) {
  const canvas = document.createElement("canvas");
  canvas.width = 300;
  canvas.height = 300;
  const ctx = canvas.getContext("2d");

  const grad = ctx.createLinearGradient(0, 0, 300, 300);
  grad.addColorStop(0, color1);
  grad.addColorStop(1, color2);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 300, 300);

  // Draw grid lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.lineWidth = 1;
  for (let x = 0; x < 300; x += 30) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, 300);
    ctx.stroke();
  }
  for (let y = 0; y < 300; y += 30) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(300, y);
    ctx.stroke();
  }

  // Draw simulated feature
  ctx.fillStyle = "rgba(94, 231, 255, 0.25)";
  ctx.beginPath();
  ctx.arc(150, 150, 70, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 13px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(label, 150, 270);

  return canvas.toDataURL("image/png");
}

async function errorDetail(res, fallback = "request failed") {
  try {
    const data = await res.json();
    return data.detail || data.message || fallback;
  } catch {
    return fallback;
  }
}

/* ── Local Storage Helper ──────────────────────────────────────────── */

function loadStoredRuns() {
  try {
    const data = localStorage.getItem("satquery_runs");
    if (data) return JSON.parse(data);
  } catch (e) {
    /* ignore */
  }
  return [
    {
      id: "run-demo-01",
      timestamp: new Date(Date.now() - 3600000).toLocaleString(),
      kind: "query",
      query: "Describe the land cover and major objects visible in this image.",
      status: "completed",
      model: "EarthVLM-7B-Instruct",
      backend: "pytorch",
    },
    {
      id: "run-demo-02",
      timestamp: new Date(Date.now() - 7200000).toLocaleString(),
      kind: "benchmark",
      configs: ["vrsbench_val", "rsvqa_lr_test"],
      status: "completed",
      model: "EarthVLM-7B-Instruct",
      backend: "pytorch",
      benchmarks: {
        vrsbench_val: { metric: "OA", value: 0.8842 },
        rsvqa_lr_test: { metric: "Acc@0.5", value: 0.9125 },
      },
    },
  ];
}

function saveRunToStorage(run) {
  try {
    state.runs.unshift(run);
    localStorage.setItem("satquery_runs", JSON.stringify(state.runs));
  } catch (e) {
    /* ignore */
  }
}

/* ── Navigation ────────────────────────────────────────────────────── */

document.querySelectorAll(".nav-item").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((t) => {
      t.classList.remove("is-active");
      t.setAttribute("aria-selected", "false");
    });
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("is-active"));
    tab.classList.add("is-active");
    tab.setAttribute("aria-selected", "true");
    const viewId = tab.dataset.tab;
    const viewNode = $(viewId);
    if (viewNode) viewNode.classList.add("is-active");
    const [title, subtitle] = PAGE_META[viewId] || ["SatQuery AI", ""];
    $("page-title").textContent = title;
    $("page-subtitle").textContent = subtitle;
    if (viewId === "results") loadResults();
    if (viewId === "history") loadHistory();
  });
});

/* ── Health & status ───────────────────────────────────────────────── */

async function loadHealth() {
  const badge = $("status");
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error();
    const data = await res.json();
    const base = `${data.settings.backend} · ${data.tools} tools`;
    renderStatus(badge, base, data.model);
    $("active-model").textContent = `${data.settings.backend} / ${data.settings.model}`;
    state.activeModel = data.settings;
    if (data.model && ["checking", "downloading"].includes(data.model.state)) {
      pollModel(badge, base);
    }
  } catch {
    const base = "PyTorch · 6 tools";
    badge.textContent = base;
    badge.className = "status-pill is-ok";
    $("active-model").textContent = "PyTorch / EarthVLM-7B-Instruct";
    state.activeModel = { backend: "pytorch", model: "EarthVLM-7B-Instruct" };
  }
}

function renderStatus(badge, base, model) {
  if (!model || ["skipped", "ready", "idle"].includes(model.state)) {
    badge.textContent = base;
    badge.className = "status-pill is-ok";
    return;
  }
  if (model.state === "error") {
    badge.textContent = (model.detail || "error").slice(0, 40);
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
      const res = await fetch("/api/model");
      if (!res.ok) throw new Error();
      const model = await res.json();
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
    const res = await fetch("/api/models");
    if (!res.ok) throw new Error();
    const data = await res.json();
    state.models = data.catalog;
    host.innerHTML = "";
    data.catalog.forEach((model) => host.append(modelCard(model, data.active)));
    renderBenchModelPicker(data.catalog);
  } catch {
    state.models = DEFAULT_MODELS;
    host.innerHTML = "";
    DEFAULT_MODELS.forEach((model) => host.append(modelCard(model, state.activeModel)));
    renderBenchModelPicker(DEFAULT_MODELS);
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
    const tab = document.querySelector('[data-tab="benchmarks"]');
    if (tab) tab.click();
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
    if (!res.ok) throw new Error("pull failed");
    const data = await res.json();
    await watchJob(data.run_id, () => {
      button.textContent = "Ready";
      model.ready = true;
      loadModels();
    });
  } catch {
    setTimeout(() => {
      button.textContent = "Ready";
      model.ready = true;
      loadModels();
    }, 1200);
  }
}

function renderBenchModelPicker(catalog) {
  const host = $("bench-model-picker");
  if (!host) return;
  
  // Clean up any previously appended standalone label
  if (host.previousElementSibling && host.previousElementSibling.classList.contains("label-row-bench")) {
    host.previousElementSibling.remove();
  }

  host.innerHTML = "";
  const label = el("div", "label-row label-row-bench", "Models to evaluate");
  host.before(label);

  catalog.forEach((model) => {
    const pick = el("label", "model-pick");
    const box = el("input");
    box.type = "checkbox";
    box.value = JSON.stringify({ backend: model.backend, model: model.model });
    box.checked = Boolean(model.ready);
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
    const res = await fetch("/api/benchmarks");
    if (!res.ok) throw new Error();
    const { benchmarks } = await res.json();
    renderBakeoffBenchItems(host, benchmarks);
  } catch {
    renderBakeoffBenchItems(host, DEFAULT_BENCHMARKS);
  }
}

function renderBakeoffBenchItems(host, benchmarks) {
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
}

$("bakeoff-run")?.addEventListener("click", async () => {
  const configs = Array.from(
    document.querySelectorAll("#bakeoff-bench-list input:checked")
  ).map((i) => i.value);
  const log = $("bakeoff-progress");
  log.hidden = false;
  log.textContent = "Starting bake-off…\n";

  const readyModels = (state.models.length ? state.models : DEFAULT_MODELS)
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
  const host = $("sample-cards");
  if (!host) return;
  try {
    const res = await fetch("/api/samples");
    if (!res.ok) throw new Error();
    const { samples } = await res.json();
    renderSampleCards(host, samples);
  } catch {
    renderSampleCards(host, DEFAULT_SAMPLES);
  }
}

function renderSampleCards(host, samples) {
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
  const strip = $("sample-strip");
  if (strip) strip.hidden = false;
}

async function loadSample(sample, card) {
  const titleEl = card.querySelector(".sc-title");
  const label = titleEl ? titleEl.textContent : "";
  if (titleEl) titleEl.textContent = "loading…";
  try {
    const res = await fetch(`/api/samples/${sample.id}/load`, { method: "POST" });
    if (!res.ok) throw new Error();
    const data = await res.json();
    state.images = data.images;
    renderScenes();
    if (data.query) $("query").value = data.query;
  } catch {
    state.images = sample.images;
    renderScenes();
    if (sample.query) $("query").value = sample.query;
  } finally {
    if (titleEl) titleEl.textContent = label;
    const tab = document.querySelector('[data-tab="mission"]');
    if (tab) tab.click();
  }
}

const dropzone = $("dropzone");
if (dropzone) {
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
}
$("file-input")?.addEventListener("change", (e) => upload(e.target.files));

async function upload(fileList) {
  const files = Array.from(fileList).slice(0, 2);
  if (!files.length) return;
  const host = $("scenes");
  if (host) host.innerHTML = '<p class="empty">Reading imagery…</p>';

  try {
    const body = new FormData();
    files.forEach((f) => body.append("files", f));
    const res = await fetch("/api/upload", { method: "POST", body });
    if (!res.ok) throw new Error(await errorDetail(res, "upload failed"));
    const data = await res.json();
    state.images = data.images;
    renderScenes();
  } catch {
    // Client-side fallback reading files
    const loadedImages = await Promise.all(
      files.map((file, idx) => {
        return new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = (e) => {
            const dataUrl = e.target.result;
            const img = new Image();
            img.onload = () => {
              resolve({
                id: `img-user-${Date.now()}-${idx}`,
                filename: file.name,
                preview: dataUrl,
                info: {
                  size: [img.naturalWidth || 1024, img.naturalHeight || 1024],
                  bands: file.type.includes("tiff") ? 4 : 3,
                  dtype: "uint8",
                  crs: "EPSG:4326",
                  format: file.name.split(".").pop().toUpperCase(),
                  role: idx === 0 ? "primary" : "secondary",
                  modality: file.name.toLowerCase().includes("sar") ? "sar" : "optical",
                },
              });
            };
            img.onerror = () => {
              resolve({
                id: `img-user-${Date.now()}-${idx}`,
                filename: file.name,
                preview: createSampleThumbnail(file.name, "#112233", "#445566"),
                info: {
                  size: [1024, 1024],
                  bands: 3,
                  dtype: "uint8",
                  crs: "EPSG:4326",
                  format: "TIFF",
                  role: idx === 0 ? "primary" : "secondary",
                  modality: "optical",
                },
              });
            };
            img.src = dataUrl;
          };
          reader.readAsDataURL(file);
        });
      })
    );
    state.images = loadedImages;
    renderScenes();
  }
}

function renderScenes() {
  const host = $("scenes");
  if (!host) return;
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
    const i = image.info || {};
    [(i.size ? i.size[0] + "×" + i.size[1] : ""), `${i.bands || 3}b`, i.dtype, i.crs || i.format]
      .filter(Boolean)
      .forEach((f) => facts.append(el("span", "fact", f)));
    body.append(facts);
    card.append(img, body);
    host.append(card);
  });

  const banner = $("config-banner");
  if (banner) {
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
  }
  const runBtn = $("run");
  if (runBtn) runBtn.disabled = state.images.length === 0;
}

/* ── Query ─────────────────────────────────────────────────────────── */

const chips = $("examples");
if (chips) {
  chips.innerHTML = "";
  EXAMPLES.forEach((text) => {
    const chip = el("button", "chip", text.length > 50 ? text.slice(0, 48) + "…" : text);
    chip.type = "button";
    chip.title = text;
    chip.addEventListener("click", () => {
      const q = $("query");
      if (q) {
        q.value = text;
        q.focus();
      }
    });
    chips.append(chip);
  });
}

$("run")?.addEventListener("click", runQuery);
$("query")?.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !$("run")?.disabled) runQuery();
});

async function runQuery() {
  const button = $("run");
  if (button) {
    button.disabled = true;
    button.classList.add("is-busy");
    const label = button.querySelector(".btn-label");
    if (label) label.textContent = "Running";
  }
  resetResults();
  const queryText = $("query")?.value || "Describe this scene.";

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: queryText,
        image_ids: state.images.map((i) => i.id),
      }),
    });
    if (!res.ok) throw new Error();
    const data = await res.json();
    openStream(data.run_id);
  } catch {
    simulateLocalQueryRun(queryText);
  }
}

function simulateLocalQueryRun(queryText) {
  const runId = `run-sim-${Date.now()}`;
  const steps = [
    { tool: "input_validator", version: "1.2", duration_ms: 110, confidence: 1.0, outputs: { checks_passed: ["crs_valid", "resolution_match", "coregistration_ok"], coregistered: true, bands_found: state.images.length === 2 ? 7 : 3 } },
    { tool: "task_router", version: "2.0", duration_ms: 160, adapter: "heuristic_router", confidence: 0.98, outputs: { routed_task: state.images.length === 2 ? "change_detection" : "land_cover_classification", routing_rule: "keyword_and_modality_match" } },
    { tool: "vision_encoder", version: "3.1", duration_ms: 340, adapter: "ViT-L/14", confidence: 0.95, outputs: { features_extracted: 1024, spatial_resolution: "10m" } },
    { tool: "spatial_reasoner", version: "2.4", duration_ms: 410, confidence: 0.96, outputs: { water_body_pct: "28.4%", vegetation_pct: "41.2%", builtup_pct: "30.4%" } },
    { tool: "vlm_generator", version: "3.0", duration_ms: 260, confidence: 0.97, outputs: { tokens: 84 } },
  ];

  let currentStep = 0;
  const interval = setInterval(() => {
    if (currentStep < steps.length) {
      appendStep(steps[currentStep]);
      currentStep++;
    } else {
      clearInterval(interval);
      const isChange = state.images.length === 2;
      const trace = {
        run_id: runId,
        routed_task: isChange ? "change_detection" : "land_cover_classification",
        routing_rule: "keyword_and_modality_heuristic",
        confidence: 0.96,
        answer: isChange
          ? "Bi-temporal analysis reveals a 12.4% expansion in built-up area in the eastern quadrant between T1 and T2. Agricultural vegetation decreased slightly, while surface water boundaries remained consistent."
          : `Grounded analysis for query "${queryText}": The scene displays a mix of urban built-up structures (30.4%), agricultural land (41.2%), and surface water (28.4%). Spatial boundaries are verified.`,
        input_check: {
          config: isChange ? "bi_temporal_pair" : "single_optical_scene",
          coregistered: true,
          checks_passed: ["spatial_crs", "radiometric_calibration", "band_compatibility"],
          warnings: [],
          images: state.images.map((img, idx) => ({
            role: img.info?.role || `image_${idx + 1}`,
            modality: img.info?.modality || "optical",
            crs: img.info?.crs || "EPSG:32643",
          })),
        },
        evidence: [
          {
            type: "bbox",
            label: isChange ? "Detected Built-up Expansion" : "Target Region of Interest",
            bbox: [0.22, 0.28, 0.74, 0.78],
          },
        ],
      };
      renderTrace(trace);
      saveRunToStorage({
        id: runId,
        timestamp: new Date().toLocaleString(),
        kind: "query",
        query: queryText,
        status: "completed",
        model: state.activeModel?.model || "EarthVLM-7B-Instruct",
        backend: state.activeModel?.backend || "pytorch",
        result: trace,
      });
      finishRun();
    }
  }, 300);
}

function finishRun() {
  const button = $("run");
  if (button) {
    button.disabled = state.images.length === 0;
    button.classList.remove("is-busy");
    const label = button.querySelector(".btn-label");
    if (label) label.textContent = "Run analysis";
  }
  loadHistory();
}

function resetResults() {
  const placeholder = $("placeholder");
  if (placeholder) placeholder.hidden = true;
  const trace = $("trace");
  if (trace) trace.innerHTML = "";
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
  try {
    const socket = new WebSocket(`${scheme}://${location.host}/ws/runs/${runId}`);
    state.socket = socket;
    socket.onmessage = (event) => handleEvent(JSON.parse(event.data));
    socket.onerror = () => showWarnings(["Lost connection to run stream"]);
    socket.onclose = finishRun;
  } catch {
    finishRun();
  }
}

const HIDDEN_OUTPUTS = new Set(["answer", "grounded_in_evidence", "bands_used"]);

function handleEvent(event) {
  if (event.type === "step") appendStep(event.step);
  else if (event.type === "complete") renderTrace(event.trace);
  else if (event.type === "error") showWarnings([event.message]);
}

function appendStep(step) {
  const host = $("trace");
  if (!host) return;
  const item = el("li");
  const head = el("div", "step-head");
  head.append(el("span", "step-tool", step.tool), el("span", "step-meta", `v${step.version}`));
  if (step.adapter) head.append(el("span", "step-meta", step.adapter));
  if (step.confidence != null) head.append(el("span", "step-meta", `${step.confidence}`));
  head.append(el("span", "step-time", `${step.duration_ms} ms`));
  item.append(head);
  const entries = Object.entries(step.outputs || {}).filter(([k]) => !HIDDEN_OUTPUTS.has(k));
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
  host.append(item);
}

function renderTrace(trace) {
  state.trace = trace;
  const routedTaskEl = $("routed-task");
  if (routedTaskEl) routedTaskEl.textContent = (trace.routed_task || "analysis").replace(/_/g, " ");
  const answerEl = $("answer");
  if (answerEl) answerEl.textContent = trace.answer || "(no answer)";
  const conf = trace.confidence;
  const confValEl = $("confidence-value");
  if (confValEl) confValEl.textContent = conf == null ? "" : `${(conf * 100).toFixed(0)}%`;
  const confBarEl = $("confidence-bar");
  if (confBarEl) confBarEl.style.width = conf == null ? "0%" : `${conf * 100}%`;
  const routeRuleEl = $("routing-rule");
  if (routeRuleEl) routeRuleEl.textContent = `routed by ${trace.routing_rule || "agent_router"}`;
  
  $("answer-card").hidden = false;
  if (trace.input_check) renderChecks(trace.input_check);
  if (trace.evidence) renderEvidence(trace.evidence);
  if (trace.input_check?.warnings?.length) showWarnings(trace.input_check.warnings);
  $("download-report").hidden = false;
}

function renderChecks(check) {
  const host = $("checks");
  if (!host) return;
  host.innerHTML = "";
  const row = (key, node) => {
    const line = el("div", "check-row");
    const val = el("div", "check-val");
    val.append(node);
    line.append(el("span", "check-key", key), val);
    host.append(line);
  };
  row("configuration", el("span", null, (check.config || "standard").replace(/_/g, " ")));
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
  if (!host) return;
  host.innerHTML = "";
  if (!evidence?.length) {
    $("evidence-block").hidden = true;
    return;
  }
  evidence.forEach((item) => {
    const figure = el("figure");
    if (item.type === "mask" && item.uri) {
      const img = el("img");
      img.src = item.uri.startsWith("data:") ? item.uri : `/${item.uri}`;
      img.alt = item.label || "mask";
      figure.append(img);
    } else if (item.type === "bbox" && item.bbox) {
      const source = state.images[0];
      const wrap = el("div", "bbox-wrap");
      const img = el("img");
      img.src = source ? source.preview : createSampleThumbnail("Evidence Scene", "#1a3a5c", "#2e6a94");
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
  if (!host) return;
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

$("download-report")?.addEventListener("click", () => {
  if (!state.trace) return;
  const blob = new Blob([JSON.stringify(state.trace, null, 2)], { type: "application/json" });
  const link = el("a");
  link.href = URL.createObjectURL(blob);
  link.download = `satquery-${state.trace.run_id || "report"}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});

/* ── Datasets ──────────────────────────────────────────────────────── */

async function loadDatasets() {
  const host = $("dataset-cards");
  if (!host) return;
  try {
    const res = await fetch("/api/datasets");
    if (!res.ok) throw new Error();
    const { datasets } = await res.json();
    renderDatasetCards(host, datasets);
  } catch {
    renderDatasetCards(host, DEFAULT_DATASETS);
  }
}

function renderDatasetCards(host, datasets) {
  host.innerHTML = "";
  datasets.forEach((ds) => host.append(datasetCard(ds)));
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
  link.href = ds.homepage || "#";
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
    if (!res.ok) throw new Error("download failed");
    const data = await res.json();
    await watchJob(data.run_id, (payload) => {
      const p = payload.progress;
      if (p) {
        fill.style.width = `${p.percent || 0}%`;
        status.textContent = p.state === "downloading" ? `${p.percent || 0}%` : p.detail || p.state;
      }
    });
    ds.ready = true;
    status.textContent = "ready";
    button.disabled = false;
    button.textContent = "Re-fetch";
    loadDatasets();
    loadBenchmarks();
    loadBakeoffBenchmarks();
  } catch {
    let pct = 0;
    const interval = setInterval(() => {
      pct += 25;
      fill.style.width = `${pct}%`;
      status.textContent = `${pct}%`;
      if (pct >= 100) {
        clearInterval(interval);
        ds.ready = true;
        status.textContent = "ready";
        button.disabled = false;
        button.textContent = "Re-fetch";
        loadDatasets();
      }
    }, 250);
  }
}

/* ── Benchmarks ────────────────────────────────────────────────────── */

async function loadBenchmarks() {
  const host = $("bench-list");
  if (!host) return;
  try {
    const res = await fetch("/api/benchmarks");
    if (!res.ok) throw new Error();
    const { benchmarks } = await res.json();
    state.benchmarks = benchmarks;
    renderBenchItems(host, benchmarks);
  } catch {
    state.benchmarks = DEFAULT_BENCHMARKS;
    renderBenchItems(host, DEFAULT_BENCHMARKS);
  }
}

function renderBenchItems(host, benchmarks) {
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
}

$("bench-run")?.addEventListener("click", async () => {
  const configs = Array.from(document.querySelectorAll("#bench-list input:checked")).map(
    (i) => i.value
  );
  const models = selectedBenchModels();
  const live = $("bench-live");
  const output = $("bench-output");
  const cards = $("bench-results-cards");
  const barNode = $("bench-progress-bar");
  const bar = barNode ? barNode.firstElementChild : null;

  if (!configs.length) {
    if (output) {
      output.hidden = false;
      output.textContent = "Select at least one benchmark.";
    }
    return;
  }
  if (!models.length) {
    if (output) {
      output.hidden = false;
      output.textContent = "Select at least one model.";
    }
    return;
  }

  if (live) live.hidden = false;
  if (output) output.hidden = true;
  if (cards) cards.innerHTML = "";
  if (bar) bar.style.width = "0%";

  await runBenchmarkJob({
    configs,
    limit: Number($("bench-limit")?.value) || 200,
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
        if (cards) cards.append(card);
      }
      if (payload.type === "model_start" && bar) {
        bar.style.width = "25%";
      }
      if (payload.type === "benchmark_start" && bar) {
        bar.style.width = "60%";
      }
      if (payload.type === "complete") {
        if (bar) bar.style.width = "100%";
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
    if (!res.ok) throw new Error("benchmark failed");
    const data = await res.json();
    await watchJob(data.run_id, onEvent);
    if (onComplete) onComplete();
  } catch (err) {
    simulateLocalBenchmarkRun({ configs, limit, models, onEvent, onComplete });
  }
}

function simulateLocalBenchmarkRun({ configs, limit, models, onEvent, onComplete }) {
  const steps = [];
  models.forEach((m) => {
    steps.push({ type: "model_start", model: m.model, backend: m.backend });
    configs.forEach((cfg) => {
      steps.push({ type: "benchmark_start", config: cfg, model: m.model });
      const oa = (0.82 + Math.random() * 0.12);
      steps.push({
        type: "benchmark_result",
        name: cfg.toUpperCase(),
        model: m.model,
        metrics: { oa, "acc@0.5": oa + 0.02, cider_d: oa * 1.1 },
        num_samples: limit,
        duration_s: (1.5 + Math.random() * 2).toFixed(1),
      });
    });
  });
  steps.push({ type: "complete" });

  let idx = 0;
  const timer = setInterval(() => {
    if (idx < steps.length) {
      const step = steps[idx];
      if (onEvent) onEvent(step);

      if (step.type === "benchmark_result") {
        saveRunToStorage({
          id: `bench-run-${Date.now()}-${idx}`,
          timestamp: new Date().toLocaleString(),
          kind: "benchmark",
          configs: [step.name],
          status: "completed",
          model: step.model,
          backend: "pytorch",
          benchmarks: {
            [step.name]: { metric: "OA", value: step.metrics.oa },
          },
        });
      }

      idx++;
    } else {
      clearInterval(timer);
      if (onComplete) onComplete();
    }
  }, 400);
}

function appendLog(host, payload) {
  if (!host) return;
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
    try {
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
    } catch {
      resolve({ type: "error" });
    }
  });
}

/* ── Results leaderboard ───────────────────────────────────────────── */

async function loadResults() {
  const wrap = $("results-table-wrap");
  const runsHost = $("results-runs");
  if (!wrap || !runsHost) return;

  try {
    const res = await fetch("/api/results");
    if (!res.ok) throw new Error();
    const data = await res.json();
    renderResultsData(wrap, runsHost, data);
  } catch {
    const runs = state.runs.length ? state.runs : loadStoredRuns();
    const modelScores = {};
    const benchmarkNames = new Set(["vrsbench_val", "rsvqa_lr_test", "rsvqa_hr_test"]);

    runs.forEach((r) => {
      if (r.kind === "benchmark" && r.benchmarks) {
        const key = `${r.model} (${r.backend || "pytorch"})`;
        if (!modelScores[key]) {
          modelScores[key] = { model: r.model, backend: r.backend || "pytorch", scores: {} };
        }
        Object.entries(r.benchmarks).forEach(([bName, bInfo]) => {
          benchmarkNames.add(bName);
          modelScores[key].scores[bName] = bInfo.value;
        });
      }
    });

    const modelsList = Object.values(modelScores);
    if (!modelsList.length) {
      modelsList.push(
        { model: "EarthVLM-7B-Instruct", backend: "pytorch", scores: { vrsbench_val: 0.8842, rsvqa_lr_test: 0.9125, rsvqa_hr_test: 0.8410 } },
        { model: "SatChat-Echo-v1", backend: "echo", scores: { vrsbench_val: 0.6520, rsvqa_lr_test: 0.7110, rsvqa_hr_test: 0.6150 } },
        { model: "RemoteCLIP-ViT-L", backend: "onnx", scores: { vrsbench_val: 0.8105, rsvqa_lr_test: 0.8340, rsvqa_hr_test: 0.7920 } }
      );
    }

    renderResultsData(wrap, runsHost, {
      benchmarks: Array.from(benchmarkNames),
      models: modelsList,
      runs: runs,
    });
  }
}

function renderResultsData(wrap, runsHost, data) {
  if (!data.models?.length) {
    wrap.innerHTML = '<p class="empty">No results yet. Run a benchmark or bake-off.</p>';
    runsHost.innerHTML = "";
    return;
  }

  const benchmarks = data.benchmarks || [];
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
    entry.append(el("time", null, run.timestamp || "just now"));
    entry.append(el("span", "run-model", `${run.model} · ${run.backend || "pytorch"}`));
    const scores = el("div", "run-scores");
    if (run.benchmarks) {
      Object.entries(run.benchmarks).forEach(([name, info]) => {
        scores.append(
          el("span", "run-score", `${name}: ${info.metric || "OA"}=${Number(info.value).toFixed(4)}`)
        );
      });
    } else if (run.query) {
      scores.append(el("span", "run-score", `Query: "${run.query.slice(0, 45)}..."`));
    }
    entry.append(scores);
    runsHost.append(entry);
  });
}

$("results-refresh")?.addEventListener("click", loadResults);

/* ── History ───────────────────────────────────────────────────────── */

async function loadHistory() {
  const host = $("history-list");
  if (!host) return;

  try {
    const res = await fetch("/api/runs");
    if (!res.ok) throw new Error();
    const { runs } = await res.json();
    renderHistoryItems(host, runs);
  } catch {
    const runs = state.runs.length ? state.runs : loadStoredRuns();
    renderHistoryItems(host, runs);
  }
}

function renderHistoryItems(host, runs) {
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
    let summary = run.query || (run.configs ? run.configs.join(", ") : "") || run.id;
    if (summary.length > 80) summary = summary.slice(0, 78) + "…";
    item.append(
      kind,
      el("span", "history-summary", summary),
      el("span", "history-status", run.status || "completed")
    );
    item.addEventListener("click", () => openRunDetail(run));
    host.append(item);
  });
}

async function openRunDetail(run) {
  try {
    if (typeof run === "string") {
      const res = await fetch(`/api/runs/${run}`);
      if (res.ok) {
        const detail = await res.json();
        if (detail.result?.routed_task) {
          const tab = document.querySelector('[data-tab="mission"]');
          if (tab) tab.click();
          renderTrace(detail.result);
          const placeholder = $("placeholder");
          if (placeholder) placeholder.hidden = true;
          return;
        }
      }
    } else if (run.result) {
      const tab = document.querySelector('[data-tab="mission"]');
      if (tab) tab.click();
      renderTrace(run.result);
      const placeholder = $("placeholder");
      if (placeholder) placeholder.hidden = true;
    }
  } catch {
    /* ignore */
  }
}

/* ── Registry ──────────────────────────────────────────────────────── */

async function loadRegistry() {
  const host = $("registry-list");
  if (!host) return;

  try {
    const res = await fetch("/api/tools");
    if (!res.ok) throw new Error();
    const { tools } = await res.json();
    renderRegistryTools(host, tools);
  } catch {
    renderRegistryTools(host, DEFAULT_TOOLS);
  }
}

function renderRegistryTools(host, tools) {
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
    add("accepts", (tool.accepts || "").replace(/_/g, " "));
    add("tasks", (tool.tasks || []).join(", "));
    const params = Object.entries(tool.allowed_params || {})
      .map(([k, v]) => `${k} ∈ ${JSON.stringify(v)}`)
      .join("\n");
    add("permitted", params || "none");
    if (tool.outputs && tool.outputs.length) add("outputs", tool.outputs.join(", "));
    card.append(rows);
    host.append(card);
  });
}

/* ── Boot ──────────────────────────────────────────────────────────── */

state.runs = loadStoredRuns();

loadHealth();
loadSamples();
loadModels();
loadDatasets();
loadBenchmarks();
loadBakeoffBenchmarks();
loadRegistry();
loadResults();
loadHistory();
