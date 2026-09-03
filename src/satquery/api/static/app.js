/* SatQuery AI — single-page client.
 *
 * No build step and no framework, deliberately: the demo has to start with one
 * command on a machine that may have no Node toolchain, and an offline venue is
 * a real risk.
 *
 * The execution trace arrives over a websocket step by step rather than as one
 * payload at the end, because watching the tools fire in order is the point.
 */

const state = { images: [], runId: null, socket: null, trace: null };

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

/* ── tabs ──────────────────────────────────────────────────── */

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $(tab.dataset.tab).classList.add("active");
  });
});

/* ── health ────────────────────────────────────────────────── */

async function loadHealth() {
  const badge = $("status");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    badge.textContent = `${data.settings.backend} · ${data.settings.model} · ${data.tools} tools`;
    badge.className = "status ok";
  } catch {
    badge.textContent = "backend unreachable";
    badge.className = "status err";
  }
}

/* ── upload ────────────────────────────────────────────────── */

const dropzone = $("dropzone");
const fileInput = $("file-input");

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("over");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("over");
  })
);
dropzone.addEventListener("drop", (e) => upload(e.dataTransfer.files));
fileInput.addEventListener("change", (e) => upload(e.target.files));

async function upload(fileList) {
  const files = Array.from(fileList).slice(0, 2);
  if (!files.length) return;

  const body = new FormData();
  files.forEach((f) => body.append("files", f));

  $("images").innerHTML = '<p class="hint">uploading…</p>';
  try {
    const res = await fetch("/api/upload", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "upload failed");
    state.images = data.images;
    renderImages();
  } catch (err) {
    $("images").innerHTML = "";
    showWarnings([String(err.message || err)]);
  }
}

function renderImages() {
  const host = $("images");
  host.innerHTML = "";
  state.images.forEach((image) => {
    const card = el("div", "thumb");
    const img = el("img");
    img.src = image.preview;
    img.alt = image.filename;
    card.append(img, el("div", "name", image.filename));

    const i = image.info;
    const facts = [
      `${i.size[0]}×${i.size[1]} · ${i.bands} band${i.bands === 1 ? "" : "s"}`,
      `${i.format} · ${i.dtype}`,
    ];
    if (i.crs) facts.push(i.crs);
    if (i.gsd_m) facts.push(`${i.gsd_m} m/px`);
    card.append(el("div", "facts", facts.join("\n")));
    host.append(card);
  });

  const badge = $("config-badge");
  if (state.images.length === 2) {
    badge.textContent = "pair uploaded — configuration is inferred at run time";
    badge.classList.remove("hidden");
  } else {
    badge.classList.add("hidden");
  }
  $("run").disabled = state.images.length === 0;
}

/* ── examples ──────────────────────────────────────────────── */

document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => {
    $("query").value = chip.textContent.trim();
  })
);

/* ── run a query ───────────────────────────────────────────── */

$("run").addEventListener("click", runQuery);

async function runQuery() {
  const button = $("run");
  button.disabled = true;
  button.textContent = "Running…";

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
    state.runId = data.run_id;
    openStream(data.run_id);
  } catch (err) {
    showWarnings([String(err.message || err)]);
    button.disabled = false;
    button.textContent = "Run analysis";
  }
}

function resetResults() {
  $("trace").innerHTML = "";
  $("evidence").innerHTML = '<p class="hint">No evidence yet.</p>';
  $("answer-card").classList.add("hidden");
  $("warnings").classList.add("hidden");
  $("download-report").classList.add("hidden");
  state.trace = null;
}

function openStream(runId) {
  if (state.socket) state.socket.close();
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/ws/runs/${runId}`);
  state.socket = socket;

  socket.onmessage = (event) => handleEvent(JSON.parse(event.data));
  socket.onerror = () => showWarnings(["lost connection to the run stream"]);
  socket.onclose = () => {
    $("run").disabled = state.images.length === 0;
    $("run").textContent = "Run analysis";
  };
}

function handleEvent(event) {
  if (event.type === "step") appendStep(event.step);
  else if (event.type === "complete") renderTrace(event.trace);
  else if (event.type === "error") showWarnings([event.message]);
}

function appendStep(step) {
  const host = $("trace");
  host.querySelector(".hint")?.remove();

  const node = el("div", "step");
  const head = el("div", "step-head");
  head.append(
    el("span", "step-num", String(step.step)),
    el("span", "step-tool", step.tool),
    el("span", "step-ver", `v${step.version}`)
  );
  if (step.adapter) head.append(el("span", "step-ver", `adapter: ${step.adapter}`));
  if (step.confidence != null)
    head.append(el("span", "step-ver", `conf ${step.confidence}`));
  head.append(el("span", "step-time", `${step.duration_ms} ms`));
  node.append(head);

  const outputs = { ...step.outputs };
  delete outputs.answer;
  if (Object.keys(outputs).length) {
    node.append(el("div", "step-out", JSON.stringify(outputs, null, 2)));
  }
  host.append(node);
}

function renderTrace(trace) {
  state.trace = trace;

  $("routed-task").textContent = trace.routed_task.replace(/_/g, " ");
  $("answer").textContent = trace.answer || "(no answer produced)";
  $("confidence").textContent =
    trace.confidence == null ? "" : `confidence ${(trace.confidence * 100).toFixed(0)}%`;
  $("routing-rule").textContent = `routed by: ${trace.routing_rule}`;
  $("answer-card").classList.remove("hidden");

  renderChecks(trace.input_check);
  renderEvidence(trace.evidence);
  showWarnings(trace.input_check.warnings || []);
  $("download-report").classList.remove("hidden");
}

function renderChecks(check) {
  const host = $("checks");
  host.innerHTML = "";

  const row = (key, valueNode) => {
    const line = el("div", "check-row");
    line.append(el("span", "check-key", key));
    line.append(valueNode);
    host.append(line);
  };

  row("configuration", el("span", null, check.config.replace(/_/g, " ")));
  row(
    "co-registered",
    el("span", check.coregistered ? "tick" : null, check.coregistered ? "yes ✓" : "not confirmed")
  );

  const passed = el("span");
  (check.checks_passed || []).forEach((c) => passed.append(el("span", "tag pass", c)));
  row("checks passed", passed);

  (check.images || []).forEach((image, index) => {
    const facts = el("span");
    facts.append(el("span", "tag", image.role));
    facts.append(el("span", "tag", image.modality));
    facts.append(el("span", "tag", `${image.size[0]}×${image.size[1]}`));
    facts.append(el("span", "tag", `${image.bands}b`));
    if (image.crs) facts.append(el("span", "tag", image.crs));
    row(`image ${index + 1}`, facts);
  });
}

function renderEvidence(evidence) {
  const host = $("evidence");
  host.innerHTML = "";
  if (!evidence || !evidence.length) {
    host.innerHTML = '<p class="hint">No visual evidence for this task.</p>';
    return;
  }

  evidence.forEach((item) => {
    const figure = el("figure");
    if (item.type === "mask" && item.uri) {
      const img = el("img");
      img.src = `/${item.uri}`;
      figure.append(img);
    } else if (item.type === "bbox" && item.bbox) {
      // Draw the box over the source image, scaled from unit coordinates.
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
}

function showWarnings(messages) {
  const host = $("warnings");
  if (!messages || !messages.length) {
    host.classList.add("hidden");
    return;
  }
  host.innerHTML = "<strong>Warnings</strong>";
  const list = el("ul");
  messages.forEach((m) => list.append(el("li", null, m)));
  host.append(list);
  host.classList.remove("hidden");
}

$("download-report").addEventListener("click", () => {
  if (!state.trace) return;
  const blob = new Blob([JSON.stringify(state.trace, null, 2)], {
    type: "application/json",
  });
  const link = el("a");
  link.href = URL.createObjectURL(blob);
  link.download = `satquery-report-${state.trace.run_id}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});

/* ── benchmarks ────────────────────────────────────────────── */

async function loadBenchmarks() {
  const host = $("bench-list");
  try {
    const res = await fetch("/api/benchmarks");
    const { benchmarks } = await res.json();
    host.innerHTML = "";
    if (!benchmarks.length) {
      host.innerHTML = '<p class="hint">No benchmark configs found.</p>';
      return;
    }
    benchmarks.forEach((bench) => {
      const row = el("div", "bench-row");
      const box = el("input");
      box.type = "checkbox";
      box.value = bench.config;
      box.checked = Boolean(bench.data_present);
      row.append(box, el("span", "name", bench.name));
      if (bench.task) row.append(el("span", "task", bench.task));
      const avail = el(
        "span",
        `avail ${bench.data_present ? "yes" : "no"}`,
        bench.error ? "config error" : bench.data_present ? "data found" : "data missing"
      );
      row.append(avail);
      host.append(row);
    });
  } catch {
    host.innerHTML = '<p class="hint">Could not load benchmarks.</p>';
  }
}

$("bench-run").addEventListener("click", async () => {
  const selected = Array.from(
    document.querySelectorAll("#bench-list input:checked")
  ).map((i) => i.value);
  const output = $("bench-output");

  if (!selected.length) {
    output.textContent = "Select at least one benchmark.";
    return;
  }

  output.textContent = "starting…\n";
  try {
    const res = await fetch("/api/benchmarks/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        configs: selected,
        limit: Number($("bench-limit").value) || 32,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "benchmark run failed");

    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${location.host}/ws/runs/${data.run_id}`);
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "benchmark_start") {
        output.textContent += `\nrunning ${payload.name}…\n`;
      } else if (payload.type === "benchmark_result") {
        output.textContent += `${payload.name} (${payload.task}) — ${payload.num_samples} samples in ${payload.duration_s}s\n`;
        Object.entries(payload.metrics)
          .filter(([k]) => !k.startsWith("n/") && k !== "n")
          .forEach(([k, v]) => {
            output.textContent += `  ${k.padEnd(20)} ${Number(v).toFixed(4)}\n`;
          });
      } else if (payload.type === "complete") {
        output.textContent += `\n${payload.result.table}\n`;
      } else if (payload.type === "error") {
        output.textContent += `\nerror: ${payload.message}\n`;
      }
    };
  } catch (err) {
    output.textContent += `\nerror: ${err.message || err}\n`;
  }
});

/* ── registry ──────────────────────────────────────────────── */

async function loadRegistry() {
  const host = $("registry-list");
  try {
    const res = await fetch("/api/tools");
    const { tools } = await res.json();
    host.innerHTML = "";
    tools.forEach((tool) => {
      const card = el("div", "tool-card");
      card.append(el("h4", null, `${tool.name} v${tool.version}`));
      card.append(el("div", "row", `accepts: ${tool.accepts.replace(/_/g, " ")}`));
      card.append(el("div", "row", `tasks: ${tool.tasks.join(", ")}`));
      const params = Object.entries(tool.allowed_params)
        .map(([k, v]) => `${k} ∈ ${JSON.stringify(v)}`)
        .join("   ");
      card.append(el("div", "row", `permitted params: ${params || "none"}`));
      if (tool.outputs.length) {
        card.append(el("div", "row", `outputs: ${tool.outputs.join(", ")}`));
      }
      host.append(card);
    });
  } catch {
    host.innerHTML = '<p class="hint">Could not load the registry.</p>';
  }
}

loadHealth();
loadBenchmarks();
loadRegistry();
