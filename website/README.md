# EntityLens

A static, interactive ground-truth explorer for multilingual business entity resolution. No model or GPU runs in the website.

## Run locally

From the repository root:

```bash
python3 -m http.server 5173 --bind 127.0.0.1 --directory website/dist
```

Open http://localhost:5173. The committed JSON snapshot makes the website independent of the pipeline, Python dependencies, and ignored fixture files.

## Explore

Search reference businesses, filter matches/singletons/missing addresses/multilingual text, select a target source, compare fields, or open the connection graph. The architecture section explains the pipeline rather than presenting an executed model trace. The initial snapshot is labelled synthetic and includes 100 reference businesses, 160 positive links, and 20 singletons.

All true matches for displayed businesses are preserved. Negative examples are targets absent from their complete ground-truth match list. Up to four negatives per target source are selected for country agreement and name similarity from a bounded sample. This selected comparison set is not the inference candidate union. Inspector text agreement is not model confidence.

## Replace the example snapshot

Use standard training TSV headers from the challenge. Supply `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, and `train_ground_truth.tsv` in one directory.

```bash
python3 website/scripts/export_examples.py \
  --data-dir /path/to/dataset/train --max-queries 100 --no-synthetic
```

The bundled fixture can be re-exported when its ignored files are present:

```bash
python3 website/scripts/export_examples.py --synthetic
```

The exporter uses only the Python standard library. Query selection uses seeded hashes. Every selected positive is retained; background targets are bounded to 2,000 per source for inexpensive comparison selection. Include only data suitable for public display when publishing.

## Free GitHub Pages deployment

GitHub Pages supports public repositories on GitHub Free. In your repository, choose **Settings → Pages → Build and deployment → Source → GitHub Actions**.

Copy `website/deploy-pages.example.yml` to `.github/workflows/deploy-pages.yml`, commit it along with the website, and push your `main` branch when you choose to publish. The workflow uploads only `website/dist/`; no build, GPU, secrets, or Python pipeline are required.

After the workflow succeeds, the site URL is:

https://niweshsah.github.io/Amazon-ML-Challenge-2026/

Relative asset links support GitHub Pages project paths. Subsequent pushes affecting the site will update the deployment. The workflow template is stored outside `.github/` so this local implementation does not publish automatically.

Reference: https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages

## Structure

`dist/` contains the deployable pages, modular styles/scripts, favicon, and labelled JSON snapshot. `scripts/` contains the data exporter. Browser WebMCP navigation is optional and feature-detected; browsers without it retain every visible interaction. Its native browser integration was not checked because no supported context was available.
