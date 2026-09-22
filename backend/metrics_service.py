"""Measured multiclass metrics and synchronized, end-to-end local inference timing."""
import time
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from backend.dataset_service import CLASSES


def classification_metrics(y_true, y_pred):
    result = {"accuracy": float(accuracy_score(y_true, y_pred)), "test_images": len(y_true)}
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(4)), average=average, zero_division=0)
        result.update({f"precision_{average}": float(p), f"recall_{average}": float(r), f"f1_{average}": float(f)})
    result["classification_report"] = classification_report(y_true, y_pred, labels=list(range(4)), target_names=CLASSES, output_dict=True, zero_division=0)
    result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(4))).tolist()
    return result


def report_frame(metrics):
    """Preserve per-class support; sklearn's scalar accuracy isn't a support count."""
    import pandas as pd
    report = metrics["classification_report"]
    rows = {name: dict(values) for name, values in report.items() if isinstance(values, dict)}
    if "accuracy" in report:
        rows["accuracy"] = {"precision": None, "recall": None, "f1-score": report["accuracy"], "support": metrics["test_images"]}
    return pd.DataFrame.from_dict(rows, orient="index")


def benchmark(backbone, head, rows, device, repeats=32):
    import torch
    from backend.model_service import image_tensor, synchronize
    head = head.to(device).eval()
    def one(row):
        x = image_tensor(row["path"]).unsqueeze(0).to(device)
        head(backbone(x))
        synchronize(device)
    timings = []
    with torch.inference_mode():
        for i in range(3):
            one(rows[i % len(rows)])
        for i in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            one(rows[i % len(rows)])
            timings.append((time.perf_counter() - start) * 1000)
    head.cpu()
    return {"latency_mean_ms": float(np.mean(timings)), "latency_median_ms": float(np.median(timings)),
            "latency_p95_ms": float(np.percentile(timings, 95)), "inference_images_per_second": 1000 / float(np.mean(timings)),
            "latency_samples_ms": timings, "benchmark_images": repeats, "warmup_images": 3,
            "benchmark_scope": "Batch size 1; file decode, resize, normalization, transfer, backbone and head; warm file cache; CUDA synchronized"}
