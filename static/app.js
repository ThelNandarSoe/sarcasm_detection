const state = {
  selectedId: window.HEADLINE_DESK.defaultModel,
  compareIds: window.HEADLINE_DESK.models.map((model) => model.id),
};

const form = document.getElementById("predict-form");
const runBtn = document.getElementById("run-btn");
const result = document.getElementById("result");
const verdict = document.getElementById("verdict");
const probability = document.getElementById("probability");
const meterFill = document.getElementById("meter-fill");
const inputPreview = document.getElementById("input-preview");
const errorBox = document.getElementById("error");
const headline = document.getElementById("headline");
const charCount = document.getElementById("char-count");
const resultModel = document.getElementById("result-model");
const resultTokenizer = document.getElementById("result-tokenizer");
const resultBlurb = document.getElementById("result-blurb");
const compareBody = document.getElementById("compare-body");
const contextFields = document.getElementById("context-fields");

function modelById(id) {
  return window.HEADLINE_DESK.models.find((item) => item.id === id);
}

function paintSelection() {
  document.querySelectorAll(".model-card").forEach((card) => {
    const selected = card.dataset.modelId === state.selectedId;
    card.classList.toggle("selected", selected);
    const mark = card.querySelector(".selected-mark");
    const button = card.querySelector(".use-model");
    if (mark) mark.hidden = !selected;
    if (button) button.textContent = selected ? "Selected" : "Use model";
  });
  const info = modelById(state.selectedId);
  if (contextFields) {
    contextFields.hidden = !info || info.family !== "transformer";
  }
}

function paintCompareChips() {
  document.querySelectorAll(".compare-chip").forEach((chip) => {
    chip.classList.toggle("active", state.compareIds.includes(chip.dataset.modelId));
  });
}

function showError(message) {
  errorBox.hidden = false;
  errorBox.textContent = message;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

headline.addEventListener("input", () => {
  charCount.textContent = `${headline.value.length} characters`;
});

document.querySelectorAll(".model-card").forEach((card) => {
  const select = () => {
    state.selectedId = card.dataset.modelId;
    paintSelection();
  };
  card.addEventListener("click", (event) => {
    if (event.target.closest("button") || event.target.closest(".use-model")) return;
    select();
  });
  card.querySelector(".use-model").addEventListener("click", select);
});

document.querySelectorAll(".chip[data-headline]").forEach((chip) => {
  chip.addEventListener("click", () => {
    headline.value = chip.dataset.headline;
    charCount.textContent = `${headline.value.length} characters`;
    headline.focus();
  });
});

document.getElementById("clear-btn").addEventListener("click", () => {
  form.reset();
  charCount.textContent = "0 characters";
  verdict.textContent = "Waiting";
  probability.textContent = "Enter a headline and run detection.";
  meterFill.style.width = "0%";
  result.classList.remove("sarcastic", "straight");
  inputPreview.hidden = true;
  resultModel.textContent = "—";
  resultTokenizer.textContent = "—";
});

document.querySelectorAll(".compare-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const id = chip.dataset.modelId;
    if (state.compareIds.includes(id)) {
      state.compareIds = state.compareIds.filter((item) => item !== id);
    } else {
      state.compareIds.push(id);
    }
    paintCompareChips();
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  runBtn.disabled = true;
  runBtn.textContent = "Checking…";
  const payload = {
    model_id: state.selectedId,
    headline: headline.value,
    author: document.getElementById("author").value,
    section: document.getElementById("section").value,
    description: document.getElementById("description").value,
  };
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");
    const sarcastic = data.predicted_label === 1;
    result.classList.toggle("sarcastic", sarcastic);
    result.classList.toggle("straight", !sarcastic);
    verdict.textContent = data.prediction;
    probability.textContent = `Sarcasm probability: ${Number(data.sarcasm_percent).toFixed(2)}%`;
    meterFill.style.width = `${Math.max(2, data.sarcasm_percent)}%`;
    resultModel.textContent = data.model_name;
    resultTokenizer.textContent = data.tokenizer || "—";
    resultBlurb.textContent = `The selected model classified this headline as ${data.prediction.toLowerCase()} based on the probability from the binary classifier.`;
    inputPreview.hidden = false;
    inputPreview.textContent = data.input_text;
  } catch (err) {
    showError(err.message);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Check headline";
  }
});

document.getElementById("compare-btn").addEventListener("click", async () => {
  clearError();
  if (!headline.value.trim()) {
    showError("Enter a headline first, then run comparison.");
    return;
  }
  if (!state.compareIds.length) {
    showError("Select at least one model to compare.");
    return;
  }
  const button = document.getElementById("compare-btn");
  button.disabled = true;
  button.textContent = "Comparing…";
  try {
    const response = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        headline: headline.value,
        model_ids: state.compareIds,
        author: document.getElementById("author").value,
        section: document.getElementById("section").value,
        description: document.getElementById("description").value,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Comparison failed.");
    if (!data.results.length) {
      compareBody.innerHTML = "<tr><td colspan='3'>No comparison results.</td></tr>";
    } else {
      compareBody.innerHTML = data.results
        .map(
          (row) =>
            `<tr><td>${row.name}</td><td>${row.prediction}</td><td>${Number(row.sarcasm_percent).toFixed(1)}%</td></tr>`
        )
        .join("");
    }
    if (data.errors?.length) {
      showError(data.errors.map((item) => `${item.model_id}: ${item.error}`).join(" "));
    }
  } catch (err) {
    showError(err.message);
  } finally {
    button.disabled = false;
    button.textContent = "Run comparison";
  }
});

paintSelection();
paintCompareChips();
