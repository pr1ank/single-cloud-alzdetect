"""Automatic orchestration, disk reuse, reproducibility, and per-run artifacts."""
from __future__ import annotations
import platform
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import pandas as pd
import torch
from filelock import FileLock

from backend.utils import ROOT, SCHEMA_VERSION, fingerprint, read_json, write_json
from backend.dataset_service import CLASSES, stratified_subset, split_dataset
from backend.training_service import train_experiment
from backend.cloud_simulation_service import simulate_aws
from backend.efficiency_service import efficiency
from backend.metrics_service import report_frame


def new_experiment_directory(root):
    root.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    for n in range(1000):
        folder = root / (base if n == 0 else f"{base}_{n:03d}")
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            continue
    raise RuntimeError("Cannot allocate experiment directory")


def save_run(folder, config, result):
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "config.json", config)
    write_json(folder / "metrics.json", result["metrics"])
    pd.DataFrame(result["history"]).to_csv(folder / "training_history.csv", index=False)
    report_frame(result["metrics"]).to_csv(folder / "classification_report.csv", index_label="class")
    pd.DataFrame(result["metrics"]["confusion_matrix"], index=CLASSES, columns=CLASSES).to_csv(folder / "confusion_matrix.csv", index_label="actual")
    pd.DataFrame(result["resource_samples"]).to_csv(folder / "resources.csv", index=False)
    pd.DataFrame({"actual": result["y_true"], "predicted": result["predictions"]}).to_csv(folder / "predictions.csv", index=False)
    torch.save(result["head"].state_dict(), folder / "classifier.pt")


def code_fingerprint():
    return fingerprint({p.name: p.read_text(encoding="utf-8") for p in sorted((ROOT / "backend").glob("*.py"))})


def run_automatic(dataset, hardware, config, pricing, model_loader, progress=lambda message: None,
                  epoch_callback=lambda row: None, root=ROOT):
    versions = {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "numpy", "scikit-learn", "streamlit")}
    identity = {"schema": SCHEMA_VERSION, "code": code_fingerprint(), "dataset": dataset["fingerprint"],
                "config": config, "cpu": hardware["cpu"], "gpu": hardware["gpu"], "versions": versions}
    key = fingerprint(identity)
    exp_root = root / "experiments"
    exp_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(exp_root / "experiment.lock"), timeout=7200):
        for path in sorted(exp_root.glob("*/summary.json"), reverse=True):
            try:
                saved = read_json(path)
                weights_retry_due = (not saved.get("model_info", {}).get("pretrained", False)
                                     and (datetime.now(timezone.utc) - datetime.fromisoformat(saved["created_at"])).total_seconds() > 3600)
                if saved.get("cache_key") == key and saved.get("status") == "complete" and not weights_retry_due:
                    progress("Complete — reusing saved measurements from " + saved["created_at"])
                    # Keep the price snapshot used by this experiment, not a silent repricing.
                    return saved
            except (OSError, ValueError):
                pass
        folder = new_experiment_directory(exp_root)
        config = dict(config)
        selected = stratified_subset(dataset["rows"], config["dataset_fraction"], config["max_images"], minimum_per_class=10)
        splits = split_dataset(selected)
        config.update(selected_images=len(selected), actual_dataset_fraction=len(selected) / dataset["valid_images"],
                      split_counts={name: len(rows) for name, rows in splits.items()}, hardware=hardware,
                      versions=versions, python=platform.python_version(), dataset_source=dataset["source"],
                      dataset_fingerprint=dataset["fingerprint"], dataset_demo=dataset["demo"], schema=SCHEMA_VERSION)
        write_json(folder / "config.json", config)
        pd.DataFrame([{**r, "split": name} for name, rows in splits.items() for r in rows]).to_csv(folder / "split_manifest.csv", index=False)
        write_json(folder / "status.json", {"status": "running", "cache_key": key})
        try:
            progress("Preparing EfficientNetB0 and public pretrained weights...")
            backbone, model_info = model_loader(config["device"])
            config["model_info"] = model_info
            write_json(folder / "config.json", config)
            progress("Training EfficientNet classification head...")
            try:
                result = train_experiment(backbone, splits, config, progress, epoch_callback)
            except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
                progress("Memory limit reached; automatically retrying on CPU with batch size 8.")
                config.update(device="cpu", batch_size=8, recovery_reason=str(exc))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                backbone, model_info = model_loader("cpu")
                config["model_info"] = model_info
                result = train_experiment(backbone, splits, config, progress, epoch_callback)
            save_run(folder, config, result)
            progress("Calculating AWS estimates and project efficiency...")
            cloud = simulate_aws(result["metrics"], pricing, dataset["size_bytes"], config["device"])
            scores = efficiency(result["metrics"], cloud)
            scaling = []
            for fraction in (.10, .25, .50, .75, 1.0):
                progress(f"Scalability experiment: {fraction:.0%} of selected training pool...")
                subfolder = folder / "scalability" / f"{int(fraction * 100):03d}"
                subconfig = {**config, "epochs": config["scalability_epochs"], "subset_fraction": fraction}
                subsplits = {**splits, "train": stratified_subset(splits["train"], fraction)}
                subconfig["split_counts"] = {name: len(rows) for name, rows in subsplits.items()}
                write_json(subfolder / "config.json", subconfig)
                try:
                    # Re-extract features and independently reset/train the head for every size.
                    sub = train_experiment(backbone, subsplits, subconfig, progress, run_benchmark=False)
                    save_run(subfolder, subconfig, sub)
                    aws = simulate_aws(sub["metrics"], pricing, dataset["size_bytes"], config["device"])
                    write_json(subfolder / "aws_estimate.json", aws)
                    m, r = sub["metrics"], sub["metrics"]["resources"]
                    scaling.append({"requested_percent": fraction * 100, "train_images": len(subsplits["train"]),
                                    "epochs": subconfig["epochs"], "training_seconds": m["training_seconds"],
                                    "accuracy": m["accuracy"], "macro_f1": m["f1_macro"],
                                    "throughput_images_s": m["training_unique_images_per_second"],
                                    "cpu_percent": r["cpu_percent"], "ram_percent": r["ram_percent"],
                                    "gpu_percent": r["gpu_percent"], "estimated_aws_usd": aws["cost"]["total_usd"], "status": "measured"})
                except Exception as exc:
                    error = {"requested_percent": fraction * 100, "status": "Experiment unavailable", "error": str(exc)}
                    scaling.append(error)
                    write_json(subfolder / "error.json", error)
            pd.DataFrame(scaling).to_csv(folder / "scalability.csv", index=False)
            summary = {"status": "complete", "cache_key": key, "created_at": datetime.now(timezone.utc).isoformat(),
                       "folder": str(folder.resolve()), "config": config, "metrics": result["metrics"],
                       "history": result["history"], "cloud": cloud, "efficiency": scores, "scalability": scaling,
                       "dataset_demo": dataset["demo"], "model_info": model_info}
            write_json(folder / "summary.json", summary)
            write_json(folder / "status.json", {"status": "complete", "cache_key": key})
            progress("Complete.")
            return summary
        except Exception as exc:
            write_json(folder / "error.json", {"status": "Experiment unavailable", "error": str(exc), "type": type(exc).__name__})
            write_json(folder / "status.json", {"status": "failed", "cache_key": key})
            raise
