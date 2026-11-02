/**
 * Dynamic Bandwidth Allocation Simulation Dashboard
 * University Project - Academic Engineering Frontend
 */

(function () {
  "use strict";

  // State
  let simState = {
    currentSlot: 8,
    totalSlots: 600,
    capacity: 100.0,
    revealed: true,
    isPlaying: false,
    timerId: null,
    historyDemand: [], // Array of [interactive, web, video, bulk]
    currentForecasts: null,
    currentAllocations: null,
    currentActual: null,
    currentErrors: null,
    runningSummary: null,
  };

  // Restrained color palette matching CSS
  const CLASS_COLORS = {
    interactive: "#1f5f8b",
    web: "#1b7a6e",
    video: "#a36417",
    bulk: "#5a6578",
  };

  const POLICY_NAMES = [
    "equal_share",
    "max_min_fairness",
    "last_observed_weighted",
    "moving_average_weighted",
    "neural_dynamic",
    "oracle",
  ];

  const POLICY_LABELS = {
    equal_share: "Equal Share",
    max_min_fairness: "Max-Min Fairness",
    last_observed_weighted: "Last-Observed + Weighted WF",
    moving_average_weighted: "Moving Average + Weighted WF",
    neural_dynamic: "Neural Dynamic + Weighted WF",
    oracle: "Oracle Reference (Ideal)",
  };

  // DOM Elements
  const el = {
    scenarioSelect: document.getElementById("scenario-select"),
    capacityInput: document.getElementById("capacity-input"),
    seedInput: document.getElementById("seed-input"),
    bufferInput: document.getElementById("buffer-input"),
    slotDurationInput: document.getElementById("slot-duration-input"),
    totalSlotsInput: document.getElementById("total-slots-input"),
    wInteractive: document.getElementById("w-interactive"),
    wWeb: document.getElementById("w-web"),
    wVideo: document.getElementById("w-video"),
    wBulk: document.getElementById("w-bulk"),
    btnReinit: document.getElementById("btn-reinit"),

    btnPlay: document.getElementById("btn-play"),
    btnPause: document.getElementById("btn-pause"),
    btnStepPredict: document.getElementById("btn-step-predict"),
    btnStepReveal: document.getElementById("btn-step-reveal"),
    btnStepBoth: document.getElementById("btn-step-both"),
    btnFastForward: document.getElementById("btn-fast-forward"),
    btnReset: document.getElementById("btn-reset"),
    speedSelect: document.getElementById("speed-select"),

    lblSlotCounter: document.getElementById("lbl-slot-counter"),
    lblStepPhase: document.getElementById("lbl-step-phase"),
    lblInspectSlot: document.getElementById("lbl-inspect-slot"),
    canvas: document.getElementById("simulation-canvas"),

    summaryTbody: document.getElementById("policy-summary-tbody"),
    classTbody: document.getElementById("class-breakdown-tbody"),
    benchmarkTbody: document.getElementById("benchmark-tbody"),
    benchmarkScenarioSelect: document.getElementById("benchmark-scenario-select"),
  };

  const ctx = el.canvas.getContext("2d");

  // Tab switching
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      document.getElementById(targetId).classList.add("active");
      if (targetId === "tab-benchmark") {
        loadBenchmarkData(el.benchmarkScenarioSelect.value);
      }
    });
  });

  el.benchmarkScenarioSelect.addEventListener("change", () => {
    loadBenchmarkData(el.benchmarkScenarioSelect.value);
  });

  // API Calls
  async function initSimulation() {
    pause();
    const payload = {
      scenario: el.scenarioSelect.value,
      capacity: parseFloat(el.capacityInput.value) || 100.0,
      seed: parseInt(el.seedInput.value, 10) || 7,
      buffer_size_mb: parseFloat(el.bufferInput.value) || 5.0,
      slot_duration_ms: parseFloat(el.slotDurationInput.value) || 50.0,
      total_slots: parseInt(el.totalSlotsInput.value, 10) || 600,
      weights: [
        parseFloat(el.wInteractive.value) || 2.2,
        parseFloat(el.wWeb.value) || 1.5,
        parseFloat(el.wVideo.value) || 1.2,
        parseFloat(el.wBulk.value) || 0.8,
      ],
    };

    try {
      const res = await fetch("/api/init", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      simState.capacity = payload.capacity;
      simState.currentSlot = data.current_slot;
      simState.totalSlots = data.total_slots;
      simState.revealed = true;
      simState.historyDemand = data.initial_history || [];
      simState.currentForecasts = null;
      simState.currentAllocations = null;
      simState.currentActual = null;
      simState.currentErrors = null;
      simState.runningSummary = null;

      updateStatusUI();
      renderInspectionTable();
      renderSummaryTable();
      renderClassTable();
      drawChart();
    } catch (err) {
      console.error("Initialization failed:", err);
    }
  }

  async function stepPredict() {
    if (simState.currentSlot >= simState.totalSlots) return;
    try {
      const res = await fetch("/api/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reveal: false }),
      });
      const data = await res.json();
      simState.revealed = false;
      simState.currentForecasts = data.forecasts;
      simState.currentAllocations = data.allocations;
      simState.currentActual = null;
      simState.currentErrors = null;

      updateStatusUI();
      renderInspectionTable();
      drawChart();
    } catch (err) {
      console.error("Step predict failed:", err);
    }
  }

  async function stepReveal() {
    try {
      const res = await fetch("/api/reveal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await res.json();
      simState.revealed = true;
      simState.currentSlot = data.next_slot;
      simState.currentActual = data.actual_demand;
      simState.currentErrors = data.forecast_errors;
      simState.runningSummary = data.summary;

      // Append to local history buffer
      simState.historyDemand.push(data.actual_demand);
      if (simState.historyDemand.length > 24) {
        simState.historyDemand.shift();
      }

      updateStatusUI();
      renderInspectionTable();
      renderSummaryTable();
      renderClassTable();
      drawChart();
    } catch (err) {
      console.error("Step reveal failed:", err);
    }
  }

  async function stepBoth() {
    if (simState.currentSlot >= simState.totalSlots) {
      pause();
      return;
    }
    if (!simState.revealed) {
      await stepReveal();
    } else {
      await stepPredict();
      await stepReveal();
    }
  }

  async function fastForwardBatch() {
    pause();
    try {
      const res = await fetch("/api/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slots: 50 }),
      });
      const data = await res.json();
      simState.currentSlot = data.next_slot;
      simState.revealed = true;
      simState.historyDemand = data.recent_history;
      simState.currentForecasts = data.last_forecasts;
      simState.currentAllocations = data.last_allocations;
      simState.currentActual = data.last_actual;
      simState.currentErrors = data.last_errors;
      simState.runningSummary = data.summary;

      updateStatusUI();
      renderInspectionTable();
      renderSummaryTable();
      renderClassTable();
      drawChart();
    } catch (err) {
      console.error("Batch advance failed:", err);
    }
  }

  // Playback control
  function play() {
    if (simState.isPlaying) return;
    simState.isPlaying = true;
    el.btnPlay.disabled = true;
    el.btnPause.disabled = false;
    const interval = parseInt(el.speedSelect.value, 10) || 250;
    simState.timerId = setInterval(async () => {
      if (simState.currentSlot >= simState.totalSlots) {
        pause();
        return;
      }
      await stepBoth();
    }, interval);
  }

  function pause() {
    simState.isPlaying = false;
    if (simState.timerId) {
      clearInterval(simState.timerId);
      simState.timerId = null;
    }
    el.btnPlay.disabled = false;
    el.btnPause.disabled = true;
  }

  el.speedSelect.addEventListener("change", () => {
    if (simState.isPlaying) {
      pause();
      play();
    }
  });

  // UI Updates
  function updateStatusUI() {
    el.lblSlotCounter.textContent = `${simState.currentSlot} / ${simState.totalSlots}`;
    el.lblInspectSlot.textContent = `Slot #${simState.currentSlot}`;

    if (!simState.revealed) {
      el.lblStepPhase.textContent = "Predicted (Waiting to Reveal)";
      el.lblStepPhase.className = "phase-badge phase-predicted";
      el.btnStepReveal.disabled = false;
      el.btnStepPredict.disabled = true;
    } else {
      el.lblStepPhase.textContent = "Revealed / Ready";
      el.lblStepPhase.className = "phase-badge phase-revealed";
      el.btnStepReveal.disabled = true;
      el.btnStepPredict.disabled = false;
    }
  }

  function renderInspectionTable() {
    const formatCell = (val) => (val !== undefined && val !== null ? `${val.toFixed(2)} Mbps` : "&mdash;");
    const formatErr = (arr) => {
      if (!arr) return "&mdash;";
      const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
      return `${mean.toFixed(2)} Mbps`;
    };

    const models = [
      { id: "row-neural", key: "neural" },
      { id: "row-ma", key: "moving_average" },
      { id: "row-last", key: "last_observed" },
      { id: "row-oracle", key: "oracle" },
    ];

    models.forEach((m) => {
      const tr = document.getElementById(m.id);
      if (!tr) return;
      const pred = simState.currentForecasts ? simState.currentForecasts[m.key] : null;
      const err = simState.currentErrors ? simState.currentErrors[m.key] : null;

      tr.querySelector(".col-c0").innerHTML = pred ? formatCell(pred[0]) : "&mdash;";
      tr.querySelector(".col-c1").innerHTML = pred ? formatCell(pred[1]) : "&mdash;";
      tr.querySelector(".col-c2").innerHTML = pred ? formatCell(pred[2]) : "&mdash;";
      tr.querySelector(".col-c3").innerHTML = pred ? formatCell(pred[3]) : "&mdash;";
      tr.querySelector(".col-err").innerHTML = formatErr(err);
    });

    // Actual row
    const actualTr = document.getElementById("row-actual");
    if (actualTr) {
      const act = simState.currentActual;
      actualTr.querySelector(".col-c0").innerHTML = act ? formatCell(act[0]) : "<em>Hidden</em>";
      actualTr.querySelector(".col-c1").innerHTML = act ? formatCell(act[1]) : "<em>Hidden</em>";
      actualTr.querySelector(".col-c2").innerHTML = act ? formatCell(act[2]) : "<em>Hidden</em>";
      actualTr.querySelector(".col-c3").innerHTML = act ? formatCell(act[3]) : "<em>Hidden</em>";
    }
  }

  function renderSummaryTable() {
    if (!simState.runningSummary) return;
    const s = simState.runningSummary;
    let html = "";

    POLICY_NAMES.forEach((p) => {
      const data = s[p];
      if (!data) return;
      const lbl = POLICY_LABELS[p] || p;
      html += `
        <tr>
          <td><strong>${lbl}</strong></td>
          <td class="num-col">${data.throughput_mbps_mean.toFixed(2)}</td>
          <td class="num-col">${data.unmet_mbps_mean.toFixed(2)}</td>
          <td class="num-col">${(data.utilization_mean * 100).toFixed(1)}%</td>
          <td class="num-col">${data.fairness_mean.toFixed(3)}</td>
          <td class="num-col">${data.backlog_mb_mean.toFixed(2)}</td>
          <td class="num-col">${data.delay_ms_mean.toFixed(1)}</td>
          <td class="num-col">${data.dropped_mbps_mean.toFixed(2)}</td>
        </tr>
      `;
    });

    el.summaryTbody.innerHTML = html;
  }

  function renderClassTable() {
    if (!simState.runningSummary) return;
    const s = simState.runningSummary;
    let html = "";

    POLICY_NAMES.forEach((p) => {
      const data = s[p];
      if (!data || !data.class_throughput_mean) return;
      const lbl = POLICY_LABELS[p] || p;
      const th = data.class_throughput_mean;
      const dl = data.class_delay_mean || {};

      html += `
        <tr>
          <td><strong>${lbl}</strong></td>
          <td class="num-col">${(th.interactive || 0).toFixed(2)}</td>
          <td class="num-col">${(th.web || 0).toFixed(2)}</td>
          <td class="num-col">${(th.video || 0).toFixed(2)}</td>
          <td class="num-col">${(th.bulk || 0).toFixed(2)}</td>
          <td class="num-col">${(dl.interactive || 0).toFixed(1)}</td>
          <td class="num-col">${(dl.bulk || 0).toFixed(1)}</td>
        </tr>
      `;
    });

    el.classTbody.innerHTML = html;
  }

  async function loadBenchmarkData(scenarioKey) {
    el.benchmarkTbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">Loading benchmark data...</td></tr>`;
    try {
      const res = await fetch("/api/benchmark");
      const allScenarios = await res.json();
      const sc = allScenarios[scenarioKey];
      if (!sc || !sc.policies) {
        el.benchmarkTbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">No benchmark data available for ${scenarioKey}. Run src/evaluate.py first.</td></tr>`;
        return;
      }

      let html = "";
      const metricCell = (mean, ci95, digits, scale = 1, suffix = "") => `
        <td class="num-col benchmark-cell">
          <span class="benchmark-mean">${(mean * scale).toFixed(digits)}${suffix}</span>
          <span class="benchmark-ci">± ${(ci95 * scale).toFixed(digits)}${suffix}</span>
        </td>
      `;

      POLICY_NAMES.forEach((p) => {
        const row = sc.policies[p];
        if (!row) return;
        const lbl = POLICY_LABELS[p] || p;
        html += `
          <tr>
            <td><strong>${lbl}</strong></td>
            ${metricCell(row.throughput_mbps_mean, row.throughput_mbps_ci95, 2)}
            ${metricCell(row.unmet_mbps_mean, row.unmet_mbps_ci95, 2)}
            ${metricCell(row.utilization_mean, row.utilization_ci95, 1, 100, "%")}
            ${metricCell(row.fairness_mean, row.fairness_ci95, 3)}
            ${metricCell(row.delay_est_ms_mean, row.delay_est_ms_ci95, 1)}
            ${metricCell(row.dropped_mbps_mean, row.dropped_mbps_ci95, 2)}
          </tr>
        `;
      });
      el.benchmarkTbody.innerHTML = html;
    } catch (err) {
      console.error("Benchmark fetch error:", err);
      el.benchmarkTbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">Error loading benchmark data.</td></tr>`;
    }
  }

  // Canvas Time-Series Chart
  function drawChart() {
    const width = el.canvas.width;
    const height = el.canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Padding
    const padLeft = 45;
    const padRight = 20;
    const padTop = 25;
    const padBottom = 30;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    const maxCapacity = simState.capacity;
    const yMax = Math.max(120.0, maxCapacity * 1.25);

    // Grid lines
    ctx.strokeStyle = "#f1f5f9";
    ctx.lineWidth = 1;
    const yTicks = [0, 25, 50, 75, 100, yMax];
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "right";

    yTicks.forEach((val) => {
      if (val > yMax) return;
      const y = padTop + plotH - (val / yMax) * plotH;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();
      ctx.fillText(`${Math.round(val)}`, padLeft - 6, y + 3);
    });

    // Draw Link Capacity dashed line
    const capY = padTop + plotH - (maxCapacity / yMax) * plotH;
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#94a3b8";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, capY);
    ctx.lineTo(width - padRight, capY);
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = "#64748b";
    ctx.textAlign = "left";
    ctx.fillText(`Capacity (${maxCapacity} Mbps)`, width - padRight - 110, capY - 5);

    // History data: array of slots
    const history = simState.historyDemand;
    if (!history || history.length === 0) return;

    const numSlots = history.length;
    const colW = plotW / Math.max(16, numSlots);

    // Stacked bars of incoming traffic
    const keys = ["interactive", "web", "video", "bulk"];

    history.forEach((slotDemand, i) => {
      const x = padLeft + i * colW + 4;
      const barW = Math.max(2, colW - 8);
      let currentBaseY = padTop + plotH;

      keys.forEach((k, classIdx) => {
        const val = slotDemand[classIdx] || 0;
        const barH = (val / yMax) * plotH;
        const barY = currentBaseY - barH;

        ctx.fillStyle = CLASS_COLORS[k];
        ctx.fillRect(x, barY, barW, barH);
        currentBaseY = barY;
      });

      // Total label for current or last slot
      if (i === numSlots - 1) {
        const total = slotDemand.reduce((a, b) => a + b, 0);
        ctx.fillStyle = "#0f172a";
        ctx.font = "bold 10px monospace";
        ctx.textAlign = "center";
        ctx.fillText(`${total.toFixed(0)}`, x + barW / 2, currentBaseY - 6);
      }
    });

    // Allocation indicator for the active slot
    if (simState.currentAllocations && simState.currentAllocations.neural_dynamic) {
      const neuralAlloc = simState.currentAllocations.neural_dynamic;
      const lastX = padLeft + (numSlots - 1) * colW + 4;
      const barW = Math.max(2, colW - 8);
      let allocBaseY = padTop + plotH;

      neuralAlloc.forEach((val, classIdx) => {
        const allocH = (val / yMax) * plotH;
        const allocY = allocBaseY - allocH;

        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(lastX, allocY, barW, allocH);

        allocBaseY = allocY;
      });
    }

    // X-axis label
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Simulated Time Slots (History &rarr; Current Slot)", width / 2, height - 10);
  }

  // Event Listeners
  el.btnPlay.addEventListener("click", play);
  el.btnPause.addEventListener("click", pause);
  el.btnStepPredict.addEventListener("click", stepPredict);
  el.btnStepReveal.addEventListener("click", stepReveal);
  el.btnStepBoth.addEventListener("click", stepBoth);
  el.btnFastForward.addEventListener("click", fastForwardBatch);
  el.btnReset.addEventListener("click", initSimulation);
  el.btnReinit.addEventListener("click", initSimulation);

  // Initialize on page load
  window.addEventListener("DOMContentLoaded", () => {
    initSimulation();
  });
})();
