"""Live plots for one dense_encoder_v2 run. Reads logs; never changes the run."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGES = ("prepare", "verify-data", "train", "build-index", "infer", "score", "validate")
COLORS = {"prepare": "#5E81AC", "verify-data": "#88C0D0", "train": "#D08770",
          "build-index": "#B48EAD", "infer": "#A3BE8C", "score": "#EBCB8B", "validate": "#81A1C1"}


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # Last line may be in the middle of a concurrent write.
            if isinstance(item, dict) and "event" in item:
                events.append(item)
    return events


def selected_run(events: list[dict], requested: str | None) -> str:
    if requested:
        return requested
    starts = [e for e in events if e.get("event") == "start" and e.get("run_id")]
    if not starts:
        raise ValueError("No run with a run_id has started yet")
    return starts[-1]["run_id"]


def timestamp(event: dict) -> datetime:
    return datetime.fromisoformat(event["timestamp_utc"].replace("Z", "+00:00"))


def style() -> None:
    plt.rcParams.update({"figure.facecolor": "#F8FAFC", "axes.facecolor": "#FFFFFF",
                         "axes.edgecolor": "#CBD5E1", "axes.labelcolor": "#334155",
                         "text.color": "#0F172A", "xtick.color": "#475569", "ytick.color": "#475569",
                         "grid.color": "#E2E8F0", "font.size": 10, "axes.titlesize": 12,
                         "axes.titleweight": "bold", "figure.titlesize": 17,
                         "savefig.dpi": 160})


def save_figure(fig, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        destination = folder / f"{name}.{extension}"
        temporary = folder / f".{name}.{extension}.tmp"
        fig.savefig(temporary, format=extension, dpi=160, bbox_inches="tight")
        os.replace(temporary, destination)
    plt.close(fig)


def pending(ax, message: str) -> None:
    ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes,
            color="#64748B", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def stage_plot(events: list[dict], folder: Path, run_id: str) -> None:
    starts = {e["command"]: e for e in events if e["event"] == "start"}
    ends = {e["command"]: e for e in events if e["event"] in ("complete", "failed")}
    first = min((timestamp(e) for e in starts.values()), default=datetime.now(timezone.utc))
    now = datetime.now(timezone.utc)
    fig, (ax, summary) = plt.subplots(2, 1, figsize=(12, 6.7),
                                      gridspec_kw={"height_ratios": [3, 1.6]})
    fig.suptitle(f"Dense encoder pipeline · {run_id}", x=.06, ha="left")
    for y, stage in enumerate(STAGES):
        if stage not in starts:
            continue
        start = (timestamp(starts[stage]) - first).total_seconds() / 60
        end_event = ends.get(stage)
        end = (timestamp(end_event) if end_event else now)
        length = max(.05, (end - timestamp(starts[stage])).total_seconds() / 60)
        ax.barh(y, length, left=start, height=.58, color=COLORS[stage], alpha=.9)
        state = "failed" if end_event and end_event["event"] == "failed" else (
            "done" if end_event else "running")
        ax.text(start + length + .1, y, f"{length:.1f} min · {state}", va="center", fontsize=9)
    ax.set_yticks(range(len(STAGES)), STAGES)
    ax.invert_yaxis()
    ax.set_xlabel("Minutes since run start")
    ax.set_title("Stage timeline", loc="left")
    ax.grid(axis="x", alpha=.7)
    ax.set_axisbelow(True)
    split = next((e for e in reversed(events) if e["event"] == "analysis_complete"), None)
    train = next((e for e in reversed(events) if e["event"] == "throughput_profile"), None)
    latest = events[-1] if events else None
    facts = [f"Latest event: {latest['command']} / {latest['event']}" if latest else "Awaiting events"]
    if split:
        facts += [f"Train pairs: {split.get('train_pairs', 0):,}     Held-out queries: {split.get('eval_groups', 0):,}",
                  f"Evaluation pool: {split.get('eval_pool_size', 0):,} targets (sampled pool)"]
    else:
        scan = next((e for e in reversed(events) if e["event"] == "split_scan_progress"), None)
        if scan:
            facts.append(f"Source 1 IDs scanned: {scan['groups_done']:,}")
    if train:
        facts.append(f"50-step training estimate: {train['estimated_train_minutes']:.1f} min at {train['max_length']} tokens")
    summary.axis("off")
    summary.text(.01, .93, "RUN SNAPSHOT", fontsize=11, weight="bold", transform=summary.transAxes)
    summary.text(.01, .76, "\n".join(facts), fontsize=10, va="top",
                 linespacing=1.65, transform=summary.transAxes)
    fig.tight_layout(rect=[0, 0, 1, .95])
    save_figure(fig, folder, "pipeline_progress")


def training_plot(events: list[dict], folder: Path, run_id: str,
                  name: str = "training_dashboard") -> None:
    steps = sorted((e for e in events if e["event"] == "train_step"), key=lambda x: x["step"])
    dev = []
    latest_checkpoint = None
    for event in events:
        if event["event"] == "checkpoint":
            latest_checkpoint = event["step"]
        elif event["event"] == "dev_retrieval" and latest_checkpoint is not None:
            dev.append((latest_checkpoint, event))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Training · {run_id}", x=.06, ha="left")
    if steps:
        x = np.array([e["step"] for e in steps])
        loss = np.array([e["loss"] for e in steps])
        axes[0, 0].plot(x, loss, color="#D08770", alpha=.35, label="Logged loss")
        window = min(7, len(loss))
        smooth = np.convolve(loss, np.ones(window) / window, mode="valid")
        axes[0, 0].plot(x[window - 1:], smooth, color="#BF616A", linewidth=2, label=f"{window}-point mean")
        axes[0, 0].legend(frameon=False)
        axes[0, 0].set_xlabel("Optimizer step")
        axes[0, 0].set_ylabel("Contrastive loss")
        axes[0, 0].set_title("In-batch + lexical-negative loss", loc="left")
        axes[0, 0].grid(alpha=.6)
        total = max(e.get("steps_total", 0) for e in steps)
        axes[0, 1].barh([0], [x[-1]], color="#D08770", height=.4)
        axes[0, 1].barh([0], [max(0, total - x[-1])], left=[x[-1]], color="#E2E8F0", height=.4)
        axes[0, 1].set_xlim(0, max(total, 1))
        axes[0, 1].set_yticks([])
        axes[0, 1].set_xlabel("Optimizer steps")
        axes[0, 1].set_title(f"Progress · {x[-1]:,} / {total:,} steps", loc="left")
        axes[1, 0].plot(x, [e.get("gpu_peak_gb", np.nan) for e in steps], color="#5E81AC", marker="o", markersize=3)
        axes[1, 0].set_xlabel("Optimizer step")
        axes[1, 0].set_ylabel("Peak reserved GPU memory (GiB)")
        axes[1, 0].set_title("GPU memory", loc="left")
        axes[1, 0].grid(alpha=.6)
    else:
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            pending(ax, "Training measurements pending")
    axes[1, 1].set_title("Development retrieval · macro F0.5", loc="left")
    if dev:
        axes[1, 1].plot([step for step, _ in dev], [e["macro_f05"] for _, e in dev],
                        color="#A3BE8C", marker="o", linewidth=2)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_xlabel("Checkpoint step")
        axes[1, 1].set_ylabel("Macro F0.5")
        axes[1, 1].grid(alpha=.6)
    else:
        pending(axes[1, 1], "First checkpoint evaluation pending")
    fig.text(.06, .015, "Development retrieval uses a separate 2,000-query split; final 80,000-query score is reported separately.",
             color="#64748B", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .94])
    save_figure(fig, folder, name)


def checkpoint_plots(events: list[dict], folder: Path, run_id: str) -> None:
    checkpoints = [(index, event) for index, event in enumerate(events)
                   if event["event"] == "checkpoint"]
    rows = []
    checkpoint_folder = folder / "checkpoints"
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    for index, event in checkpoints:
        step = event["step"]
        cutoff = next((j for j in range(index + 1, len(events))
                       if events[j]["event"] == "train_step" and events[j]["step"] > step),
                      len(events))
        snapshot = events[:cutoff]
        name = f"checkpoint_{step:05d}"
        # Regenerate the latest snapshot while its development evaluation may still arrive.
        if not (checkpoint_folder / f"{name}.png").exists() or (index, event) == checkpoints[-1]:
            training_plot(snapshot, checkpoint_folder, run_id, name)
        last_loss = next((e for e in reversed(snapshot) if e["event"] == "train_step"), None)
        dev = next((e for e in snapshot[index + 1:] if e["event"] == "dev_retrieval"), None)
        rows.append({"step": step, "timestamp_utc": event["timestamp_utc"],
                     "loss": last_loss.get("loss") if last_loss else None,
                     "peak_gpu_gb": last_loss.get("gpu_peak_gb") if last_loss else None,
                     "dev_macro_f0.5": dev.get("macro_f05") if dev else None})
    if rows:
        import csv
        destination = checkpoint_folder / "checkpoint_summary.csv"
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def validation_plot(metrics: dict | None, folder: Path, run_id: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Validation · sampled-pool · {run_id}", x=.06, ha="left")
    if not metrics:
        for ax in axes.flat:
            pending(ax, "Validation pending: inference and scoring have not completed")
        fig.tight_layout(rect=[0, 0, 1, .94])
        save_figure(fig, folder, "validation_dashboard")
        return
    holdout = metrics["heldout_score"]
    calibration = metrics["calibration"]
    full = metrics["full_100k_diagnostics"]
    k_values = [1, 5, 10, 20]
    recall = [holdout["recall_at"].get(str(k), np.nan) for k in k_values]
    axes[0, 0].bar([str(k) for k in k_values], recall, color="#5E81AC")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Queries with any true match retrieved")
    axes[0, 0].set_xlabel("Top K")
    axes[0, 0].set_title("Held-out Recall@K", loc="left")
    curve = metrics.get("threshold_curve")
    if curve:
        axes[0, 1].plot([p["threshold"] for p in curve], [p["macro_f0.5"] for p in curve], color="#B48EAD", linewidth=2)
        axes[0, 1].axvline(metrics["threshold"], color="#BF616A", linestyle="--",
                           label=f"Selected {metrics['threshold']:.2f}")
        axes[0, 1].legend(frameon=False)
        axes[0, 1].set_xlabel("Cosine similarity threshold")
        axes[0, 1].set_ylabel("Calibration macro F0.5")
        axes[0, 1].set_title("20,000-query threshold calibration", loc="left")
    else:
        pending(axes[0, 1], f"Selected threshold: {metrics['threshold']:.2f}\nCalibration curve unavailable")
    labels = ["Calibration\n20k", "Held-out score\n80k", "Full diagnostics\n100k"]
    parts = [calibration, holdout, full]
    positions = np.arange(3)
    width = .24
    for offset, key, label, color in [(-width, "macro_f0.5", "Macro F0.5", "#D08770"),
                                      (0, "precision", "Pair precision", "#A3BE8C"),
                                      (width, "recall", "Pair recall", "#5E81AC")]:
        axes[1, 0].bar(positions + offset, [p[key] for p in parts], width=width, label=label, color=color)
    axes[1, 0].set_xticks(positions, labels)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Matching quality", loc="left")
    axes[1, 0].legend(frameon=False, fontsize=8)
    preferred = ["cross_script", "mixed_script", "devanagari", "bengali", "tamil", "telugu",
                 "missing_field", "short_text", "near_duplicate", "singleton"]
    groups = full.get("difficult_subsets", {})
    shown = [(tag, groups[tag]) for tag in preferred if tag in groups]
    if shown:
        names = [f"{tag} ({value['queries']:,})" for tag, value in shown]
        values = [value["macro_f0.5"] for _, value in shown]
        axes[1, 1].barh(names[::-1], values[::-1], color="#88C0D0")
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_xlabel("Macro F0.5")
        axes[1, 1].set_title("Difficult subsets · all 100k", loc="left")
    else:
        pending(axes[1, 1], "Subset results unavailable")
    fig.text(.06, .015,
             f"Pool: {metrics['pool_size']:,} targets. Threshold chosen only on calibration; held-out score uses separate 80,000 queries. Not a full-source score.",
             color="#64748B", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .94])
    save_figure(fig, folder, "validation_dashboard")


def generate(root: Path, run_id: str | None = None) -> tuple[str, bool]:
    events = read_events(root / "logs/events.jsonl")
    run_id = selected_run(events, run_id)
    current = [e for e in events if e.get("run_id") == run_id]
    if not current:
        raise ValueError(f"No events for run {run_id}")
    folder = root / "plots" / run_id
    style()
    stage_plot(current, folder, run_id)
    training_plot(current, folder, run_id)
    checkpoint_plots(current, folder, run_id)
    score_completed = any(e["command"] == "score" and e["event"] == "complete" for e in current)
    metrics_path = root / "metrics.json"
    report = json.loads(metrics_path.read_text()) if score_completed and metrics_path.exists() else None
    validation_plot(report, folder, run_id)
    failed = next((e for e in reversed(current) if e["event"] == "failed"), None)
    done = any(e["command"] == "validate" and e["event"] == "complete" for e in current)
    latest = current[-1]
    lines = [f"# Dense encoder run {run_id}", "", f"Updated: {datetime.now(timezone.utc).isoformat()}",
             f"Latest event: `{latest['command']}` / `{latest['event']}`", "",
             "## Figures", "", "- `pipeline_progress.png` — stage timing and current progress",
             "- `training_dashboard.png` — loss, step progress, GPU memory, development F0.5",
             "- `checkpoints/checkpoint_<step>.png` — training curves saved at every checkpoint",
             "- `checkpoints/checkpoint_summary.csv` — checkpoint loss, memory, and development score",
             "- `validation_dashboard.png` — retrieval and matching results when scoring completes", ""]
    if report:
        result = report["heldout_score"]
        lines += ["## Sampled-pool held-out score", "",
                  f"Pool size: {report['pool_size']:,} targets; held-out queries: {result['queries']:,}.",
                  f"Threshold: {report['threshold']:.3f}; macro F0.5: {result['macro_f0.5']:.4f}; "
                  f"precision: {result['precision']:.4f}; recall: {result['recall']:.4f}.", ""]
    else:
        lines += ["Validation is pending; no measured final score is available yet.", ""]
    if failed:
        lines += [f"Run failed during `{failed['command']}`: `{failed.get('error', 'unknown error')}`.", ""]
    (folder / "status.md").write_text("\n".join(lines), encoding="utf-8")
    return run_id, bool(done or failed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--run-id")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    while True:
        try:
            run_id, finished = generate(args.root, args.run_id)
            print(f"{datetime.now(timezone.utc).isoformat()} plots updated for {run_id}", flush=True)
        except Exception as error:
            print(f"{datetime.now(timezone.utc).isoformat()} plotting error: {error!r}", flush=True)
            if not args.watch:
                raise
            finished = False
        if not args.watch or finished:
            break
        time.sleep(max(10, args.interval))


if __name__ == "__main__":
    main()
