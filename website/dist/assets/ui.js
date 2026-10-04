/** Shared text and record rendering, with escaped dataset values. */
export function escapeHtml(value = "") {
  return String(value).replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
}

export function normalize(value) {
  return value.normalize("NFKC").toLocaleLowerCase().replace(/\s+/gu, " ").trim();
}

export function recordSummary(example) {
  const record = example.record;
  const count = example.match_count;
  return `<div class="record-overline"><div><span class="source-pill">REFERENCE · SOURCE 1</span><span class="record-id">${escapeHtml(record.entity_id)}</span></div><span class="match-badge ${count ? "positive" : "neutral"}">${count ? `✓ ${count} true matches` : "No true matches"}</span></div>
    <h3>${escapeHtml(record.business_name) || "Unnamed business"}</h3>
    <div class="record-fields"><span><span class="field-symbol" aria-hidden="true">⌖</span>${escapeHtml(record.business_address) || "Address not provided"}</span><span>${escapeHtml(record.country) || "Country not provided"}</span></div>`;
}

export function renderSummary(metadata) {
  const cards = [
    ["Reference businesses", metadata.reference_count, "Source 1 examples"],
    ["True connections", metadata.true_links, "Across Sources 2 and 3"],
    ["Businesses with no matches", metadata.singletons, "Singletons are valid outcomes"],
    ["Countries represented", metadata.countries.length, metadata.countries.join(" · ")]
  ];
  document.querySelector("#summary").innerHTML = cards.map(([label, value, note]) =>
    `<div class="stat-card"><span>${label}</span><strong>${value.toLocaleString()}</strong><small>${escapeHtml(note)}</small></div>`).join("");
  document.querySelector("#dataset-title").textContent = metadata.title;
  document.querySelector("#reference-count").textContent = `${metadata.reference_count} records`;
}

export function comparisonCard(comparison) {
  const record = comparison.target;
  const status = comparison.is_match ? "positive" : "negative";
  return `<article class="comparison-card ${status}"><div><div class="comparison-labels"><span class="source-label">SOURCE ${comparison.source}</span><span class="record-id">${escapeHtml(record.entity_id)}</span><span class="match-badge ${status}">${comparison.is_match ? "✓ True match" : "× Negative example"}</span></div><h4>${escapeHtml(record.business_name) || "Unnamed business"}</h4><p>${escapeHtml(record.business_address) || "Address not provided"} · ${escapeHtml(record.country)}</p></div><button class="compare-button" data-target="${escapeHtml(record.entity_id)}" aria-label="Compare with ${escapeHtml(record.entity_id)}">Compare records</button></article>`;
}

export function renderComparisons(example, source) {
  const comparisons = example.comparisons.filter(comparison => source === "all" || comparison.source === Number(source));
  const matchCount = comparisons.filter(comparison => comparison.is_match).length;
  document.querySelector("#comparison-count").textContent = `${matchCount} matches · ${comparisons.length - matchCount} negatives`;
  const singletonNote = example.match_count ? "" : `<div class="singleton-message"><strong>This business is a singleton.</strong> Ground truth lists no matches in either target source. The records below are negative examples.</div>`;
  document.querySelector("#comparison-list").innerHTML = singletonNote + comparisons.map(comparisonCard).join("");
}
