/** Responsive record graph using DOM nodes and SVG connector geometry. */
import { escapeHtml } from "./ui.js";

let resizeObserver;

function drawConnections() {
  const container = document.querySelector("#connection-graph");
  const root = container.querySelector(".graph-root");
  const lines = container.querySelector(".graph-lines");
  if (!root || !lines || !container.offsetWidth) return;
  const canvasBounds = container.getBoundingClientRect();
  const rootBounds = root.getBoundingClientRect();
  const startX = rootBounds.right - canvasBounds.left;
  const startY = rootBounds.top + rootBounds.height / 2 - canvasBounds.top;
  lines.setAttribute("viewBox", `0 0 ${canvasBounds.width} ${canvasBounds.height}`);
  lines.innerHTML = [...container.querySelectorAll(".graph-node")].map(node => {
    const bounds = node.getBoundingClientRect();
    const endX = bounds.left - canvasBounds.left;
    const endY = bounds.top + bounds.height / 2 - canvasBounds.top;
    const midpoint = (startX + endX) / 2;
    return `<path class="${node.classList.contains("negative") ? "negative" : "positive"}" d="M ${startX} ${startY} C ${midpoint} ${startY}, ${midpoint} ${endY}, ${endX} ${endY}"/>`;
  }).join("");
}

export function renderGraph(example, showNegatives) {
  const container = document.querySelector("#connection-graph");
  const matches = example.comparisons.filter(item => item.is_match);
  const negatives = showNegatives ? [2, 3].flatMap(source => example.comparisons.filter(item => !item.is_match && item.source === source).slice(0, 1)) : [];
  const comparisons = [...matches, ...negatives];
  const record = example.record;
  container.innerHTML = `<svg class="graph-lines" aria-hidden="true"></svg>
    <div class="graph-root"><span class="source-pill">SOURCE 1</span><strong>${escapeHtml(record.business_name)}</strong><span class="record-id">${escapeHtml(record.entity_id)}</span></div>
    <div class="graph-targets">${comparisons.map(comparison => `<button class="graph-node ${comparison.is_match ? "positive" : "negative"}" data-target="${escapeHtml(comparison.target.entity_id)}" aria-label="Inspect ${escapeHtml(comparison.target.entity_id)}, ${comparison.is_match ? "true match" : "negative example"}"><span class="match-badge ${comparison.is_match ? "positive" : "negative"}">${comparison.is_match ? "✓ True match" : "× Negative"}</span><strong>${escapeHtml(comparison.target.business_name)}</strong><small>Source ${comparison.source} · ${escapeHtml(comparison.target.entity_id)}</small></button>`).join("") || '<p class="empty-state"><strong>No true connections.</strong>This business is a singleton.</p>'}</div>`;
  resizeObserver?.disconnect();
  if ("ResizeObserver" in window) {
    resizeObserver = new ResizeObserver(drawConnections);
    resizeObserver.observe(container);
  }
  requestAnimationFrame(drawConnections);
}
