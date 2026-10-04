/** Explorer state and user actions; all outcomes come from the exported labels. */
import { escapeHtml, normalize, recordSummary, renderSummary, renderComparisons } from "./ui.js";
import { initializeInspector, openInspector } from "./inspector.js";
import { renderGraph } from "./graph.js";
import { initializeArchitecture } from "./architecture.js";
import { registerExplorerTools } from "./tools.js";

const state = { examples: [], selectedId: null, search: "", filter: "all", source: "all", view: "comparisons" };

function selectedExample() {
  return state.examples.find(example => example.record.entity_id === state.selectedId);
}

function filteredExamples() {
  const search = normalize(state.search);
  return state.examples.filter(example => {
    const record = example.record;
    const searchableText = normalize(Object.values(record).join(" "));
    if (!searchableText.includes(search)) return false;
    if (state.filter === "matched") return example.match_count > 0;
    if (state.filter === "singleton") return example.match_count === 0;
    if (state.filter === "missing") return !record.business_address.trim();
    if (state.filter === "multilingual") return /[^\u0000-\u007f]/u.test(record.business_name);
    return true;
  });
}

function renderBusinessList() {
  const examples = filteredExamples();
  document.querySelector("#visible-count").textContent = examples.length;
  document.querySelector("#business-list").innerHTML = examples.map(example => {
    const record = example.record;
    const selected = record.entity_id === state.selectedId;
    return `<button class="business-item ${selected ? "selected" : ""}" data-query="${escapeHtml(record.entity_id)}" aria-pressed="${selected}"><span class="item-name">${escapeHtml(record.business_name) || "Unnamed business"}</span><span class="item-country">${escapeHtml(record.country)}</span><span class="item-meta"><span>${escapeHtml(record.entity_id)}</span><span class="item-match-count ${example.match_count ? "" : "zero"}">${example.match_count ? `${example.match_count} matches` : "No matches"}</span></span></button>`;
  }).join("") || '<p class="empty-state" role="status"><strong>No records found.</strong>Try another search or filter.</p>';
}

function renderSelection() {
  const example = selectedExample();
  if (!example) {
    document.querySelector("#selected-record").innerHTML = '<p class="empty-state">No business selected. Change your search or filter.</p>';
    document.querySelector("#comparison-list").innerHTML = "";
    document.querySelector("#comparison-count").textContent = "";
    document.querySelector("#connection-graph").innerHTML = '<p class="empty-state">No business selected.</p>';
    return;
  }
  document.querySelector("#selected-record").innerHTML = recordSummary(example);
  renderComparisons(example, state.source);
  if (state.view === "graph") renderGraph(example, document.querySelector("#show-negatives").checked);
}

function updateFilteredSelection() {
  const available = filteredExamples();
  if (!available.some(example => example.record.entity_id === state.selectedId)) state.selectedId = available[0]?.record.entity_id ?? null;
  renderBusinessList();
  renderSelection();
}

function selectBusiness(queryId) {
  if (!state.examples.some(example => example.record.entity_id === queryId)) throw new Error("Unknown reference business.");
  state.selectedId = queryId;
  renderBusinessList();
  renderSelection();
}

function switchView(view) {
  state.view = view;
  document.querySelectorAll("[data-view]").forEach(button => {
    const selected = button.dataset.view === view;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  document.querySelector("#comparisons-view").hidden = view !== "comparisons";
  document.querySelector("#graph-view").hidden = view !== "graph";
  renderSelection();
}

function bindActions() {
  document.querySelector("#search").addEventListener("input", event => {
    state.search = event.target.value;
    updateFilteredSelection();
  });
  document.querySelector("#filters").addEventListener("click", event => {
    const button = event.target.closest("[data-filter]");
    if (!button) return;
    state.filter = button.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach(item => {
      item.classList.toggle("selected", item === button);
      item.setAttribute("aria-pressed", String(item === button));
    });
    updateFilteredSelection();
  });
  document.querySelector("#business-list").addEventListener("click", event => {
    const button = event.target.closest("[data-query]");
    if (button) selectBusiness(button.dataset.query);
  });
  document.querySelector("#source-filters").addEventListener("click", event => {
    const button = event.target.closest("[data-source]");
    if (!button) return;
    state.source = button.dataset.source;
    document.querySelectorAll("[data-source]").forEach(item => {
      item.classList.toggle("selected", item === button);
      item.setAttribute("aria-pressed", String(item === button));
    });
    renderSelection();
  });
  document.querySelector(".view-toolbar").addEventListener("click", event => {
    const button = event.target.closest("[data-view]");
    if (button) switchView(button.dataset.view);
  });
  document.querySelector("[role=tablist]").addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const view = event.key === "Home" ? "comparisons" : event.key === "End" ? "graph" : state.view === "graph" ? "comparisons" : "graph";
    switchView(view);
    document.querySelector(`[data-view="${view}"]`).focus();
  });
  for (const selector of ["#comparison-list", "#connection-graph"]) document.querySelector(selector).addEventListener("click", event => {
    const button = event.target.closest("[data-target]");
    if (button && selectedExample()) openInspector(selectedExample(), button.dataset.target);
  });
  document.querySelector("#show-negatives").addEventListener("change", renderSelection);
}

async function initialize() {
  initializeArchitecture();
  initializeInspector();
  bindActions();
  try {
    const response = await fetch("assets/examples.json");
    if (!response.ok) throw new Error("Example data could not be loaded.");
    const snapshot = await response.json();
    if (!Array.isArray(snapshot.examples)) throw new Error("The example snapshot has an invalid format.");
    state.examples = snapshot.examples;
    state.selectedId = state.examples.find(example => example.match_count && /[^\u0000-\u007f]/u.test(example.record.business_name))?.record.entity_id ?? state.examples[0]?.record.entity_id;
    renderSummary(snapshot.metadata);
    renderBusinessList();
    renderSelection();
    registerExplorerTools({ getExamples: () => state.examples, selectBusiness, getSelected: selectedExample, openInspector });
  } catch (error) {
    document.querySelector("#dataset-title").textContent = "Example data unavailable";
    document.querySelector("#business-list").innerHTML = `<p class="empty-state" role="alert"><strong>Couldn’t load the examples.</strong>${escapeHtml(error.message)} Reload the page or check the exported snapshot.</p>`;
  }
}
initialize();
