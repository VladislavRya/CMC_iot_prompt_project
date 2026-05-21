#!/usr/bin/env python3
"""
Run all classifiers on the same data split and write comparison tables (JSON + Markdown).

IMPORTANT — how metrics are computed:
  Subprocess (each malware_classifier.py main) is for demo/logs only.
  Metrics in comparison*.md/json come from evaluate_in_process(), which:
    1. loads data via src.data_loader (same CSV, same features for all)
    2. uses the same train/test split (random_state=42)
    3. calls build_pipeline(X_train) from each script

  If LLM scripts produce the same algorithm (RF + similar preprocessing),
  metrics will match — that is expected, not a script bug.
  Differences appear only when build_pipeline() differs materially.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.data_loader import find_csv, load_dataset  # noqa: E402

DATA_DIR = ROOT / "data" / "raw"
if not list(DATA_DIR.glob("*.csv")):
    raise FileNotFoundError(f"No CSV files in {DATA_DIR}")

SPLIT_DESC = "80/20 stratified, random_state=42"

RUNNERS = [
    {
        "id": "openai_gpt55_en",
        "label": "GPT-5.5 (EN prompt)",
        "script": ROOT / "generated" / "openai_gpt55_en" / "malware_classifier.py",
        "prompt": "prompts/prompt_en.md",
    },
    {
        "id": "openai_gpt55_ru",
        "label": "GPT-5.5 (RU prompt)",
        "script": ROOT / "generated" / "openai_gpt55_ru" / "malware_classifier.py",
        "prompt": "prompts/prompt_ru.md",
    },
    {
        "id": "anthropic_opus46_en",
        "label": "Anthropic Claude 4.6 Opus (EN prompt)",
        "script": ROOT / "generated" / "anthropic_opus46_en" / "malware_classifier.py",
        "prompt": "prompts/prompt_en.md",
    },
    {
        "id": "anthropic_opus46_ru",
        "label": "Anthropic Claude 4.6 Opus (RU prompt)",
        "script": ROOT / "generated" / "anthropic_opus46_ru" / "malware_classifier.py",
        "prompt": "prompts/prompt_ru.md",
    },
    {
        "id": "yandex_alice_en",
        "label": "Yandex Alice AI (EN prompt)",
        "script": ROOT / "generated" / "yandex_alice_en" / "malware_classifier.py",
        "prompt": "prompts/prompt_en.md",
    },
    {
        "id": "yandex_alice_ru",
        "label": "Yandex Alice AI (RU prompt)",
        "script": ROOT / "generated" / "yandex_alice_ru" / "malware_classifier.py",
        "prompt": "prompts/prompt_ru.md",
    },
    {
        "id": "hand_coded",
        "label": "Hand-coded baseline",
        "script": ROOT / "baseline" / "hand_coded_classifier.py",
        "prompt": "N/A (expert implementation)",
    },
]


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def build_pipeline_from_module(mod, X_train):
    if hasattr(mod, "build_pipeline"):
        return mod.build_pipeline(X_train)
    if hasattr(mod, "create_model_pipeline"):
        return mod.create_model_pipeline(X_train)
    if hasattr(mod, "построить_конвейер"):
        return mod.построить_конвейер(X_train)
    if hasattr(mod, "build_baseline_pipeline"):
        if len(X_train) > 100_000 and hasattr(mod, "build_rf_baseline"):
            return mod.build_rf_baseline(X_train)
        return mod.build_baseline_pipeline(X_train)
    raise AttributeError("No pipeline builder in module")


def evaluate_in_process(script: Path) -> dict:
    """Unified evaluation: train + test metrics on the same fixed split."""
    module_name = f"runner_{script.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise ImportError(script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    X, y = load_dataset(DATA_DIR)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline_from_module(mod, X_train)
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    return {
        "train_size": len(y_train),
        "test_size": len(y_test),
        "train": compute_metrics(y_train, train_pred),
        "test": compute_metrics(y_test, test_pred),
        "test_predictions": test_pred,
    }


def run_subprocess(script: Path) -> int:
    return subprocess.run(
        [sys.executable, str(script), "--data-dir", str(DATA_DIR)],
        cwd=ROOT,
        check=False,
    ).returncode


def write_markdown_table(
    path: Path,
    title: str,
    subtitle: str,
    rows: list[dict],
    metrics_key: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        subtitle,
        "",
        "| Model | Prompt | Accuracy | Precision | Recall | F1 |",
        "|-------|--------|----------|-----------|--------|-----|",
    ]
    for r in rows:
        m = r[metrics_key]
        lines.append(
            f"| {r['label']} | `{r['prompt']}` | "
            f"{m['accuracy']:.6f} | {m['precision']:.6f} | "
            f"{m['recall']:.6f} | {m['f1']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    csv_used = find_csv(DATA_DIR)
    X, y = load_dataset(DATA_DIR)
    _, y_test = train_test_split(y, test_size=0.2, random_state=42, stratify=y)

    rows = []
    prev_test_pred = None
    for runner in RUNNERS:
        print(f"\n>>> {runner['label']}")
        rc = run_subprocess(runner["script"])
        eval_result = evaluate_in_process(runner["script"])
        test_pred = eval_result.pop("test_predictions")

        if prev_test_pred is not None:
            n_diff = int((prev_test_pred != test_pred).sum())
            if n_diff == 0:
                print(
                    "    ⚠ test predictions identical to previous runner "
                    "(same pipeline effect on unified data)"
                )
            else:
                print(f"    test predictions differ from previous: {n_diff} rows")
        prev_test_pred = test_pred

        rows.append(
            {
                "id": runner["id"],
                "label": runner["label"],
                "script": str(runner["script"]),
                "prompt": runner["prompt"],
                "train_size": eval_result["train_size"],
                "test_size": eval_result["test_size"],
                "metrics_train": eval_result["train"],
                "metrics_test": eval_result["test"],
                "subprocess_exit": rc,
            }
        )
        tr, te = eval_result["train"], eval_result["test"]
        print(
            f"    train F1={tr['f1']:.4f}  test F1={te['f1']:.4f}  "
            f"(n_train={eval_result['train_size']}, n_test={eval_result['test_size']})"
        )

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(csv_used),
        "dataset_name": csv_used.name,
        "total_rows": len(X),
        "split": SPLIT_DESC,
        "train_rows": eval_result["train_size"] if rows else None,
        "test_rows": eval_result["test_size"] if rows else None,
    }

    train_report = {**meta, "split_part": "train", "results": rows}
    test_report = {**meta, "split_part": "test", "results": rows}

    for report, suffix, key in (
        (train_report, "", "metrics_train"),
        (test_report, "_test", "metrics_test"),
    ):
        out_json = ROOT / "results" / f"comparison{suffix}.json"
        # JSON: flatten metrics under "metrics" for backward-friendly reading
        serializable = {
            **{k: v for k, v in report.items() if k != "results"},
            "results": [
                {
                    "id": r["id"],
                    "label": r["label"],
                    "script": r["script"],
                    "prompt": r["prompt"],
                    "metrics": r[key],
                    "subprocess_exit": r["subprocess_exit"],
                }
                for r in rows
            ],
        }
        out_json.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    subtitle = (
        f"Dataset: `{csv_used.name}` · {SPLIT_DESC} · "
        f"train n={rows[0]['train_size'] if rows else '—'}"
    )
    write_markdown_table(
        ROOT / "results" / "comparison.md",
        "Comparison table (train)",
        subtitle,
        [{**r, "metrics_train": r["metrics_train"]} for r in rows],
        "metrics_train",
    )

    subtitle_test = (
        f"Dataset: `{csv_used.name}` · {SPLIT_DESC} · "
        f"**test holdout** n={rows[0]['test_size'] if rows else '—'}"
    )
    write_markdown_table(
        ROOT / "results" / "comparison_test.md",
        "Comparison table (test)",
        subtitle_test,
        [{**r, "metrics_test": r["metrics_test"]} for r in rows],
        "metrics_test",
    )

    print(f"\nWrote {ROOT / 'results' / 'comparison.json'}")
    print(f"Wrote {ROOT / 'results' / 'comparison.md'} (train)")
    print(f"Wrote {ROOT / 'results' / 'comparison_test.json'}")
    print(f"Wrote {ROOT / 'results' / 'comparison_test.md'} (test)")


if __name__ == "__main__":
    main()
