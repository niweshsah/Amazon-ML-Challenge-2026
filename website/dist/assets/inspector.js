/** Field-by-field ground-truth inspection. No matching model is called. */
import { escapeHtml, normalize } from "./ui.js";

function highlightDifferences(value, otherValue) {
  if (!value) return '<span class="missing-field">Not provided</span>';
  const otherTokens = new Set(normalize(otherValue).split(/\s+/u));
  return value.split(/(\s+)/u).map(token => {
    const escaped = escapeHtml(token);
    return token.trim() && !otherTokens.has(normalize(token)) ? `<mark>${escaped}</mark>` : escaped;
  }).join("");
}

function agreementLabel(left, right) {
  if (!left || !right) return ["Missing information", ""];
  if (normalize(left) === normalize(right)) return ["Same normalized text", "exact"];
  const leftTokens = new Set(normalize(left).split(/\s+/u));
  const overlap = normalize(right).split(/\s+/u).some(token => leftTokens.has(token));
  return [overlap ? "Some shared text" : "Different text", ""];
}

export function openInspector(example, targetId) {
  const comparison = example.comparisons.find(item => item.target.entity_id === targetId);
  if (!comparison) throw new Error("The requested comparison is not in this example.");
  const status = comparison.is_match ? "positive" : "negative";
  const fieldRows = [["business_name", "Name"], ["business_address", "Address"], ["country", "Country"]].map(([field, label]) => {
    const left = example.record[field];
    const right = comparison.target[field];
    const [agreement, className] = agreementLabel(left, right);
    return `<tr><td>${label}</td><td>${highlightDifferences(left, right)}</td><td>${highlightDifferences(right, left)}<br><span class="agreement-chip ${className}">${agreement}</span></td></tr>`;
  }).join("");
  document.querySelector("#dialog-content").innerHTML = `
    <div class="dialog-title-row"><div><h2 id="dialog-title">Same business or a similar record?</h2><p class="dialog-description">Highlighted words differ between the two records.</p></div><span class="match-badge ${status}">${comparison.is_match ? "✓ True match" : "× Negative example"}</span></div>
    <table class="comparison-table"><thead><tr><th scope="col">Field</th><th scope="col">Source 1 · reference<br>${escapeHtml(example.record.entity_id)}</th><th scope="col">Source ${comparison.source} · comparison<br>${escapeHtml(targetId)}</th></tr></thead><tbody>${fieldRows}</tbody></table>
    <div class="inspector-note">${comparison.is_match ? "This target ID appears in the reference business’s ground-truth match list." : "This target ID is absent from the reference business’s complete ground-truth match list. Similar names or addresses alone do not establish identity."} Text agreement describes the fields; it is not a model confidence score.</div>`;
  const dialog = document.querySelector("#comparison-dialog");
  if (!dialog.open) dialog.showModal();
}

export function initializeInspector() {
  const dialog = document.querySelector("#comparison-dialog");
  document.querySelector("#close-dialog").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  });
}
