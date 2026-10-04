/** Explanatory pipeline stages, independent of example labels. */
import { escapeHtml } from "./ui.js";

const stages = [
  { title: "Preserve & prepare", subtitle: "Unicode-safe records", tags: ["Raw neural text", "Normalized lexical views"], description: "Keep original names, addresses, and country strings. Build separate normalized and transliterated views for lexical comparisons without erasing Indic combining marks.", example: "Name: श्री गणेश दुकान 1\nAddress: 201 Rue du Marché\nCountry: India" },
  { title: "Retrieve candidates", subtitle: "Two independent channels", tags: ["Indic BGE-M3", "Character-trigram TF-IDF"], description: "Search by semantic embeddings and character-level text similarity independently in each target source. The initial configuration retrieves up to 150 records per channel per source.", example: "Source 2: dense Top-150 + lexical Top-150\nSource 3: dense Top-150 + lexical Top-150\nRetrieval does not inject known true matches." },
  { title: "Union & compare", subtitle: "One deduplicated pair set", tags: ["Channel provenance", "Field-level features"], description: "Union and deduplicate the candidates, preserving channel scores and ranks. Compare names, addresses, country, missingness, and address numbers. Entity IDs and labels stay out of model features.", example: "Dense candidates ∪ TF-IDF candidates\nName agreement · Address agreement\nCountry agreement · Missing fields · Numbers" },
  { title: "Score & route", subtitle: "A matching cascade", tags: ["XGBoost", "Supervised BGE reranker"], description: "XGBoost rejects low-scoring pairs and accepts high-scoring pairs. Intermediate pairs go to the supervised cross-encoder. Thresholds are calibrated on held-out groups.", example: "Below lower threshold: reject\nAbove upper threshold: accept\nBetween thresholds: cross-encoder review" },
  { title: "Resolve identities", subtitle: "Zero, one, or many matches", tags: ["Independent pair decisions", "Macro F0.5 evaluation"], description: "Preserve every accepted pair and allow an empty match list. Validate complete Source 1 coverage and ensure every output match belongs to the candidate set. Evaluate singletons as well as matching businesses.", example: "S1-0001 → S2-0001, S3-0001\nS1-0000 → no matches\nThese examples show labels, not predictions." }
];

export function selectStage(index) {
  if (!Number.isInteger(index) || index < 0 || index >= stages.length) throw new Error("Unknown pipeline stage.");
  document.querySelectorAll(".pipeline-stage").forEach((button, buttonIndex) => {
    button.classList.toggle("selected", index === buttonIndex);
    button.setAttribute("aria-pressed", String(index === buttonIndex));
  });
  const stage = stages[index];
  document.querySelector("#stage-detail").innerHTML = `<div><h3>${stage.title}</h3><p>${stage.description}</p><div class="stage-tags">${stage.tags.map(tag => `<span>${tag}</span>`).join("")}</div></div><div class="stage-example"><div class="eyebrow">${index === 0 || index === 4 ? "ILLUSTRATIVE EXAMPLE" : "HOW THIS STAGE WORKS"}</div><code>${escapeHtml(stage.example)}</code></div>`;
}

export function initializeArchitecture() {
  document.querySelector("#pipeline-stages").innerHTML = stages.map((stage, index) => `<button class="pipeline-stage" data-stage="${index}" aria-pressed="${index === 0}"><span class="stage-number">0${index + 1}</span><strong>${stage.title}</strong><span>${stage.subtitle}</span></button>`).join("");
  document.querySelector("#pipeline-stages").addEventListener("click", event => {
    const button = event.target.closest("[data-stage]");
    if (button) selectStage(Number(button.dataset.stage));
  });
  selectStage(0);
}
