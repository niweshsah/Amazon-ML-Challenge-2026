"""Keep the root README and static architecture page aligned with one source."""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "website/docs/architecture.json"

FLOWCHART = """flowchart TD
    Inputs[Source 1 reference + Source 2/3 targets] --> Prepare[Raw Unicode + normalized/transliterated views]
    Prepare --> Dense[Indic BGE-M3 + LoRA / normalized CLS]
    Prepare --> Lexical[Character-trigram TF-IDF]
    Dense --> Search[FAISS exact or IVF-PQ / Top-150 per source]
    Lexical --> Sparse[Bounded sparse search / Top-150 per source]
    Search --> Union[Deduplicate candidate union / keep scores and ranks]
    Sparse --> Union
    Union --> Candidates[candidate_pairs.tsv / exact scored set]
    Union --> Features[22 pair features / IDs and labels excluded]
    Features --> Tree[XGBoost + isotonic calibration]
    Tree --> Low{Calibrated pair score}
    Low -->|Below lower threshold| Reject[Reject]
    Low -->|Above upper threshold| Accept[Accept]
    Low -->|Between thresholds inclusive| Cross[BGE reranker + LoRA / joint pair scoring]
    Cross --> Calibration[Cross-encoder isotonic calibration]
    Calibration --> Threshold{At least acceptance threshold?}
    Threshold -->|Yes| Accept
    Threshold -->|No| Reject
    Accept --> Output[matching_results.tsv / zero, one, or many matches]
    Output --> Validate[Validate coverage, unique IDs, candidate containment]
    Truth[Training labels / selected links and singletons] --> Groups[70% fit / 10% dev / 10% calibration / 10% audit]
    Groups -.->|Supervised fitting and dev selection| Dense
    Groups -.->|Supervised fitting and dev selection| Tree
    Groups -.->|Supervised fitting and dev selection| Cross
    Groups -.->|Two disjoint calibration halves| Calibration
    Groups -.->|Held-out audit| Evaluation[Macro F0.5 + retrieval recall + routing + subgroups]
    Output --> Evaluation"""


def build_readme(document):
    lines = [
        f"# {document['name']}", "",
        f"**[Open the live project explorer]({document['live_url']})** · "
        "[Pipeline setup](final_pipeline/README.md) · [Website setup](website/README.md)", "",
        "Resolve multilingual business records across three independent sources with "
        "Indic BGE-M3 retrieval, character-trigram TF-IDF, XGBoost, and a supervised cross-encoder.", "",
        "The live explorer uses labelled synthetic examples. Model training and full inference "
        "use the separate server environment; the website itself runs no models.", "",
        "## Complete architecture", "", "```mermaid", FLOWCHART, "```", "",
    ]
    for section in document["sections"]:
        lines.extend([f"## {section['title']}", "", section["intro"], ""])
        lines.extend(f"- {point}" for point in section["points"])
        lines.append("")
    lines.extend([
        "## Start locally", "", "```bash",
        "python3 -m http.server 5173 --bind 127.0.0.1 --directory website/dist", "```", "",
        "Open http://localhost:5173. See the linked setup guides for model training, "
        "fixture verification, data export, and deployment.", "",
        "Architecture content is maintained in `website/docs/architecture.json`. "
        "Run `python3 website/scripts/build_architecture_docs.py` after editing it to regenerate "
        "this README and the website’s full architecture page.", "",
    ])
    (ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def architecture_overview():
    return """<section class="architecture-overview" aria-labelledby="flow-title">
      <div class="eyebrow">INFERENCE FLOW</div>
      <h2 id="flow-title">Two retrieval paths. One matching cascade.</h2>
      <div class="flow-step"><span class="flow-index">01</span><div><strong>Raw records and text views</strong><p>Source 1 reference · Source 2/3 targets · Raw Unicode · Lexical views</p></div></div>
      <div class="flow-channels">
        <div class="flow-channel"><span class="eyebrow">DENSE CHANNEL</span><h3>Indic BGE-M3 + LoRA</h3><p>Normalized CLS embeddings · FAISS exact / IVF-PQ · Top-150 per source</p></div>
        <div class="flow-channel"><span class="eyebrow">LEXICAL CHANNEL</span><h3>Character-trigram TF-IDF</h3><p>Normalized + transliterated text · Batched sparse search · Top-150 per source</p></div>
      </div>
      <div class="flow-step"><span class="flow-index">02</span><div><strong>Candidate union and 22 pair features</strong><p>Deduplicate pairs · Preserve scores, ranks, and channel provenance · Export the exact scored candidate set</p></div></div>
      <div class="flow-step scoring-step"><span class="flow-index">03</span><div><strong>XGBoost and score calibration</strong><p>Field agreement · Missingness · Address numbers · Calibrated pair scores</p></div></div>
      <div class="routing-branches">
        <div class="routing-branch reject"><span>BELOW LOWER THRESHOLD</span><strong>Reject pair</strong></div>
        <div class="routing-branch review"><span>BETWEEN THRESHOLDS</span><strong>BGE cross-encoder</strong><p>Joint pair scoring · Isotonic calibration · Acceptance threshold</p></div>
        <div class="routing-branch accept"><span>ABOVE UPPER THRESHOLD</span><strong>Accept pair</strong></div>
      </div>
      <div class="flow-step"><span class="flow-index">04</span><div><strong>Independent final matches and validated TSVs</strong><p>Empty, single, or multiple matches · Complete test coverage · Every match contained in its candidate set</p></div></div>
      <div class="training-lane"><div><span class="eyebrow">SEPARATE TRAINING PATH</span><p>Seeded entity groups · Held-out positive targets excluded from fitting</p></div><div class="split-grid"><span><strong>70%</strong>Fitting</span><span><strong>10%</strong>Development</span><span><strong>10%</strong>Calibration</span><span><strong>10%</strong>Audit</span></div></div>
    </section>"""


def build_page(document):
    escape = html.escape
    navigation = "".join(f'<a href="#{section["id"]}">{escape(section["title"])}</a>' for section in document["sections"])
    sections = []
    for section in document["sections"]:
        points = "\n".join(f"          <li>{escape(point)}</li>" for point in section["points"])
        sections.append(f'''      <section class="architecture-chapter" id="{section['id']}">
        <h2>{escape(section['title'])}</h2>
        <p>{escape(section['intro'])}</p>
        <ul>
{points}
        </ul>
      </section>''')
    page = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="The complete EntityLens architecture: Unicode preprocessing, hybrid retrieval, 22 pair features, supervised matching, calibration, evaluation, and deployment.">
  <title>Full Architecture · {escape(document['name'])}</title>
  <link rel="canonical" href="{document['live_url']}architecture.html">
  <link rel="icon" type="image/svg+xml" href="assets/favicon.svg">
  <link rel="stylesheet" href="assets/base.css">
  <link rel="stylesheet" href="assets/architecture-page.css">
</head>
<body>
  <a class="skip-link" href="#architecture-content">Skip to architecture</a>
  <header class="site-header">
    <a class="brand" href="index.html"><span class="brand-mark" aria-hidden="true">e<span>l</span></span>EntityLens</a>
    <nav aria-label="Main navigation"><a class="nav-link" href="index.html">Explorer</a><a class="nav-link active" href="architecture.html">Architecture</a><a class="nav-link" href="methodology.html">Methodology</a></nav>
    <a class="repository-link" href="https://github.com/niweshsah/Amazon-ML-Challenge-2026" target="_blank" rel="noopener noreferrer">View GitHub</a>
  </header>
  <main class="page-shell architecture-page" id="architecture-content">
    <div class="eyebrow">FULL SYSTEM ARCHITECTURE</div>
    <h1>Inside <span>EntityLens.</span></h1>
    <p class="architecture-intro">Multilingual Business Entity Resolution — from raw records to calibrated matching decisions.</p>
    <p class="architecture-notice">This page documents the runnable pipeline. The explorer displays labelled ground-truth examples; no model runs in your browser.</p>
    {architecture_overview()}
    <nav class="architecture-contents" aria-label="Architecture sections">{navigation}</nav>
    <div class="architecture-chapters">
{chr(10).join(sections)}
    </div>
    <footer><span><strong>EntityLens</strong> · Multilingual Business Entity Resolution</span><div><a href="index.html">Explore examples</a><a href="methodology.html">Methodology</a></div></footer>
  </main>
</body>
</html>
'''
    (ROOT / "website/dist/architecture.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    architecture = json.loads(SOURCE.read_text(encoding="utf-8"))
    build_readme(architecture)
    build_page(architecture)
    print("Updated root README and static architecture page from the shared content source.")
