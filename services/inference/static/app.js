const MODEL_DISPLAY_NAMES = {
  majority_class: "Approve-all policy",
  logistic_regression: "Traditional scorecard",
};

const FIELD_LABELS = {
  loan_amount: "Requested loan amount (USD)",
  annual_income: "Annual income (USD)",
  debt_to_income_ratio: "Debt-to-income ratio",
  credit_score: "Credit score",
  interest_rate: "Interest rate (APR)",
  installment: "Monthly installment (USD)",
  revol_util: "Revolving credit utilisation",
  grade: "Loan grade",
  term: "Loan term",
  loan_purpose: "Loan purpose",
};

const FIELD_HINTS = {
  debt_to_income_ratio: "Enter as decimal: 0.28 = 28%",
  interest_rate: "Enter as decimal: 0.118 = 11.8%",
  revol_util: "Enter as decimal: 0.55 = 55%",
  grade: "A = lowest risk band, G = highest",
};

const form = document.getElementById("applicationForm");
const formFields = document.getElementById("formFields");
const submitBtn = document.getElementById("submitBtn");
const loadSampleBtn = document.getElementById("loadSampleBtn");
const resultEmpty = document.getElementById("resultEmpty");
const resultContent = document.getElementById("resultContent");
const errorAlert = document.getElementById("errorAlert");
const resultSubtitle = document.getElementById("resultSubtitle");

let applicationSchema = null;
let preprocessingAvailable = false;
let modelAvailable = false;

function formatFieldLabel(name) {
  return FIELD_LABELS[name] ?? name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatOptionLabel(value) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function buildNumberInput(name, meta, sampleValue) {
  const input = document.createElement("input");
  input.type = "number";
  input.name = name;
  input.required = Boolean(meta.required);

  if (meta.type === "integer") {
    input.step = "1";
  } else if (name === "interest_rate" || name === "revol_util" || name === "debt_to_income_ratio") {
    input.step = "0.001";
  } else {
    input.step = "any";
  }

  if (meta.minimum !== undefined) input.min = String(meta.minimum);
  if (meta.maximum !== undefined) input.max = String(meta.maximum);
  if (meta.exclusiveMinimum !== undefined) {
    input.min = String(meta.exclusiveMinimum);
  }

  if (sampleValue !== undefined) {
    input.placeholder = String(sampleValue);
  }

  return input;
}

function buildSelectInput(name, meta) {
  const select = document.createElement("select");
  select.name = name;
  select.required = Boolean(meta.required);
  select.innerHTML = [
    '<option value="" disabled selected>Select…</option>',
    ...(meta.options ?? []).map(
      (option) => `<option value="${option}">${formatOptionLabel(option)}</option>`
    ),
  ].join("");
  return select;
}

function buildFormFields(schema) {
  formFields.innerHTML = "";
  const sample = schema.sample_application ?? {};

  for (const name of schema.features ?? Object.keys(schema.fields)) {
    const meta = schema.fields[name];
    if (!meta) continue;

    const label = document.createElement("label");
    label.textContent = formatFieldLabel(name);

    const control =
      meta.options?.length > 0
        ? buildSelectInput(name, meta)
        : buildNumberInput(name, meta, sample[name]);

    label.appendChild(control);

    if (FIELD_HINTS[name]) {
      const hint = document.createElement("span");
      hint.className = "field-hint";
      hint.textContent = FIELD_HINTS[name];
      label.appendChild(hint);
    }

    if (meta.description) {
      label.title = meta.description;
    }

    formFields.appendChild(label);
  }
}

function formatModelName(name) {
  if (!name) return "—";
  if (name.startsWith("XGBoost")) {
    const versionMatch = name.match(/v(\d+)/);
    return versionMatch
      ? `Production risk model (v${versionMatch[1]})`
      : "Production risk model";
  }
  return MODEL_DISPLAY_NAMES[name] ?? name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatPrimaryRead(isPaidBack) {
  return isPaidBack ? "Favor payback" : "Elevated default risk";
}

function getRiskTier(defaultProbability) {
  if (defaultProbability >= 0.5) {
    return {
      tier: "High",
      headline: "Refer for enhanced review",
      summary:
        "Default risk is above the 50% policy line. Manual underwriting review is recommended before any approval.",
    };
  }
  if (defaultProbability >= 0.25) {
    return {
      tier: "Medium",
      headline: "Standard underwriting review",
      summary:
        "Risk is between policy bands. Use alongside existing credit policy and analyst judgment.",
    };
  }
  return {
    tier: "Low",
    headline: "Lower relative default risk",
    summary:
      "Estimated payback likelihood is stronger on this profile. Still subject to full policy and compliance checks.",
  };
}

function truncateText(ctx, text, maxWidth) {
  if (ctx.measureText(text).width <= maxWidth) {
    return text;
  }
  let trimmed = text;
  while (trimmed.length > 1 && ctx.measureText(`${trimmed}…`).width > maxWidth) {
    trimmed = trimmed.slice(0, -1);
  }
  return `${trimmed}…`;
}

function drawBenchmarkChart(rows) {
  const canvas = document.getElementById("benchmarkChart");
  if (!canvas) return;

  const sorted = [...rows].sort((a, b) => (b.val_auc ?? 0) - (a.val_auc ?? 0));
  const rowCount = sorted.length;
  const dpr = window.devicePixelRatio || 1;
  const rowPitch = 52;
  const barHeight = 16;
  const width = canvas.parentElement?.clientWidth || 860;
  const padding = { top: 36, right: 52, bottom: 32, left: 16 };
  const height = padding.top + padding.bottom + rowCount * rowPitch;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  if (!rowCount) {
    ctx.fillStyle = "#6b6b6b";
    ctx.font = "14px system-ui, sans-serif";
    ctx.fillText("Validation results unavailable", padding.left, height / 2);
    return;
  }

  const colors = { bar: "#2a2a2a", grid: "#e5e5e0", text: "#6b6b6b", label: "#1a1a1a" };
  const labels = sorted.map((row) => formatModelName(row.name));

  ctx.font = "12px system-ui, sans-serif";
  const labelColumnWidth = Math.min(
    Math.max(...labels.map((label) => ctx.measureText(label).width)) + 8,
    240
  );
  const plotGutter = 20;
  const plotLeft = padding.left + labelColumnWidth + plotGutter;
  const plotWidth = Math.max(width - plotLeft - padding.right, 120);
  const labelRight = padding.left + labelColumnWidth;

  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  for (let tick = 0; tick <= 5; tick += 1) {
    const x = plotLeft + (plotWidth / 5) * tick;
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, height - padding.bottom);
    ctx.stroke();

    ctx.fillStyle = colors.text;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText((tick / 5).toFixed(1), x, height - padding.bottom + 18);
  }

  ctx.beginPath();
  ctx.moveTo(labelRight + plotGutter / 2, padding.top - 6);
  ctx.lineTo(labelRight + plotGutter / 2, height - padding.bottom);
  ctx.stroke();

  ctx.fillStyle = colors.text;
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Risk ranking quality (validation set)", plotLeft + plotWidth / 2, 18);

  sorted.forEach((row, index) => {
    const yCenter = padding.top + index * rowPitch + rowPitch / 2;
    const label = truncateText(ctx, labels[index], labelColumnWidth - 4);
    const score = row.val_auc ?? 0;
    const barWidth = Math.max(score * plotWidth, score > 0 ? 2 : 0);

    ctx.fillStyle = colors.label;
    ctx.font = "12px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(label, labelRight, yCenter);

    ctx.fillStyle = colors.bar;
    ctx.fillRect(plotLeft, yCenter - barHeight / 2, barWidth, barHeight);

    ctx.fillStyle = colors.text;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(score.toFixed(3), plotLeft + barWidth + 6, yCenter);
  });
}

function showError(message) {
  errorAlert.textContent = message;
  errorAlert.classList.remove("hidden");
}

function clearError() {
  errorAlert.textContent = "";
  errorAlert.classList.add("hidden");
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function updatePolicyThreshold(threshold) {
  const el = document.getElementById("policyThreshold");
  if (el && threshold !== undefined) {
    el.textContent = `${(threshold * 100).toFixed(0)}% payback probability`;
  }
}

function renderResult(data) {
  const isPaidBack = data.prediction === 1;
  const defaultProbability = data.default_probability ?? (1 - data.confidence);
  const paidBackProbability = data.paid_back_probability ?? data.confidence;
  const risk = getRiskTier(defaultProbability);

  resultEmpty.classList.add("hidden");
  resultContent.classList.remove("hidden");
  clearError();

  document.getElementById("riskTier").textContent = `${risk.tier} risk`;
  document.getElementById("riskHeadline").textContent = risk.headline;
  document.getElementById("riskSummary").textContent = risk.summary;
  document.getElementById("paidBackProbability").textContent =
    formatPercent(paidBackProbability);
  document.getElementById("defaultProbability").textContent =
    formatPercent(defaultProbability);
  document.getElementById("predictionLabel").textContent = formatPrimaryRead(isPaidBack);
  document.getElementById("probabilityFill").style.width =
    `${paidBackProbability * 100}%`;

  resultSubtitle.textContent =
    "Illustrative read only — combine with credit policy, capacity, and compliance review.";
}

function readFormData() {
  const formData = new FormData(form);
  const payload = {};

  for (const name of applicationSchema?.features ?? []) {
    const meta = applicationSchema.fields[name];
    const raw = formData.get(name);
    if (raw === null || raw === "") continue;

    if (meta?.options || meta?.type === "string") {
      payload[name] = raw;
    } else if (meta?.type === "integer") {
      payload[name] = Number.parseInt(raw, 10);
    } else {
      payload[name] = Number(raw);
    }
  }

  return payload;
}

function fillForm(application) {
  Object.entries(application).forEach(([key, value]) => {
    if (form.elements[key]) {
      form.elements[key].value = value;
    }
  });
}

async function loadSchema() {
  const response = await fetch("/schema");
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Could not load application schema.");
  }
  applicationSchema = data;
  buildFormFields(data);
  updatePolicyThreshold(data.prediction_threshold);
  if (data.sample_application) {
    fillForm(data.sample_application);
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/health_check");
    if (!response.ok) {
      throw new Error("Health check failed");
    }
    const data = await response.json();
    preprocessingAvailable = Boolean(data.preprocessing_available);
    modelAvailable = Boolean(data.model_loaded);
  } catch (error) {
    preprocessingAvailable = false;
    modelAvailable = false;
  }
}

let benchmarkRows = [];

async function refreshBaselines() {
  const baselinesSubtitle = document.getElementById("baselinesSubtitle");

  try {
    const response = await fetch("/results");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Could not load benchmarks.");
    }

    if (!data.baselines?.length && !data.xgboost_model) {
      baselinesSubtitle.textContent = "Validation results are not available yet.";
      benchmarkRows = [];
      drawBenchmarkChart(benchmarkRows);
      return;
    }

    benchmarkRows = [...(data.baselines || [])];
    if (data.xgboost_model) {
      benchmarkRows.push(data.xgboost_model);
    }

    baselinesSubtitle.textContent =
      "Sorted by risk ranking quality on the validation sample (higher = better separation of payback vs default).";
    drawBenchmarkChart(benchmarkRows);
  } catch (error) {
    baselinesSubtitle.textContent = "Validation results could not be loaded.";
    benchmarkRows = [];
    drawBenchmarkChart(benchmarkRows);
  }
}

async function submitApplication(event) {
  event.preventDefault();
  clearError();
  submitBtn.disabled = true;
  submitBtn.textContent = "Scoring…";

  const payload = readFormData();

  try {
    if (!modelAvailable) {
      throw new Error(
        "The scoring service is not ready yet. Please try again in a moment."
      );
    }
    if (!preprocessingAvailable) {
      throw new Error(
        "Scoring is temporarily unavailable. Please try again later."
      );
    }

    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail ?? "Could not score this application. Check the inputs and try again.");
    }

    renderResult(data);
  } catch (error) {
    showError(error.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Score application";
  }
}

async function init() {
  try {
    await loadSchema();
  } catch (error) {
    showError(error.message);
  }
  await checkHealth();
  await refreshBaselines();
}

init();
window.addEventListener("resize", () => {
  if (benchmarkRows.length) {
    drawBenchmarkChart(benchmarkRows);
  }
});
form.addEventListener("submit", submitApplication);
loadSampleBtn.addEventListener("click", () => {
  if (applicationSchema?.sample_application) {
    fillForm(applicationSchema.sample_application);
  }
});
