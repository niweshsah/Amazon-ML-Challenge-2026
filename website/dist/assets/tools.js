/** Optional browser-agent navigation, using the same actions as the interface. */
export function registerExplorerTools(actions) {
  const context = document.modelContext;
  if (!context?.registerTool) return;
  const lifecycle = new AbortController();
  const tools = [
    {
      name: "list_reference_businesses", description: "Read reference IDs, names, and ground-truth match counts available in EntityLens.",
      inputSchema: { type: "object", properties: {}, additionalProperties: false },
      annotations: { readOnlyHint: true, untrustedContentHint: true },
      execute(input) {
        if (!input || Object.keys(input).length) throw new Error("No parameters are accepted.");
        return actions.getExamples().map(example => ({ id: example.record.entity_id, name: example.record.business_name, matches: example.match_count }));
      }
    },
    {
      name: "select_reference_business", description: "Select a reference business and update the visible comparisons.",
      inputSchema: { type: "object", properties: { queryId: { type: "string" } }, required: ["queryId"], additionalProperties: false },
      annotations: { readOnlyHint: false, untrustedContentHint: true },
      execute(input) {
        if (!input || typeof input.queryId !== "string" || Object.keys(input).some(key => key !== "queryId")) throw new Error("Provide a queryId string.");
        actions.selectBusiness(input.queryId);
        return { selected: actions.getSelected().record.entity_id, matches: actions.getSelected().match_count };
      }
    }
  ];
  for (const tool of tools) {
    try { Promise.resolve(context.registerTool(tool, { signal: lifecycle.signal })).catch(() => {}); }
    catch { /* Browser tool support is optional; visible interactions still work. */ }
  }
  window.addEventListener("pagehide", () => lifecycle.abort(), { once: true });
}
