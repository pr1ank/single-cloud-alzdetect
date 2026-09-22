"""Frozen-backbone transfer learning; validation selects head, test remains held out."""
import copy
import time
import numpy as np
import torch
from torch import nn
from backend.dataset_service import CLASSES
from backend.model_service import extract_features, make_head, seed_everything, synchronize
from backend.monitoring_service import ResourceMonitor
from backend.metrics_service import classification_metrics, benchmark


def train_experiment(backbone, splits, config, progress=lambda message: None, epoch_callback=lambda row: None, run_benchmark=True):
    seed_everything(config["seed"])
    device, batch_size = config["device"], config["batch_size"]
    labels = {name: torch.tensor([CLASSES.index(r["class"]) for r in rows]) for name, rows in splits.items()}
    head = make_head()
    history, best_loss, best_state = [], float("inf"), None
    synchronize(device)
    start = time.perf_counter()
    with ResourceMonitor(device) as monitor:
        features = {name: extract_features(backbone, splits[name], batch_size, device, progress) for name in ("train", "validation")}
        extraction_seconds = time.perf_counter() - start
        counts = torch.bincount(labels["train"], minlength=4).clamp_min(1).float()
        loss_fn = nn.CrossEntropyLoss(weight=counts.sum() / (4 * counts))
        optimizer = torch.optim.Adam(head.parameters(), lr=config["learning_rate"])
        for epoch in range(config["epochs"]):
            epoch_start = time.perf_counter()
            head.train()
            order = torch.randperm(len(features["train"]))
            for offset in range(0, len(order), batch_size):
                idx = order[offset:offset + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(head(features["train"][idx]), labels["train"][idx])
                loss.backward()
                optimizer.step()
            head.eval()
            with torch.no_grad():
                train_logits = head(features["train"])
                val_logits = head(features["validation"])
                val_loss = float(loss_fn(val_logits, labels["validation"]))
                row = {"epoch": epoch + 1, "loss": float(loss_fn(train_logits, labels["train"])),
                       "accuracy": float((train_logits.argmax(1) == labels["train"]).float().mean()),
                       "val_loss": val_loss, "val_accuracy": float((val_logits.argmax(1) == labels["validation"]).float().mean()),
                       "epoch_seconds": time.perf_counter() - epoch_start}
            history.append(row)
            epoch_callback(row)
            progress(f"Epoch {epoch + 1}/{config['epochs']} — loss {row['loss']:.4f}, val accuracy {row['val_accuracy']:.1%}")
            if val_loss < best_loss:
                best_loss, best_state = val_loss, copy.deepcopy(head.state_dict())
        synchronize(device)
        training_seconds = time.perf_counter() - start
    head.load_state_dict(best_state)
    head.eval()
    progress("Evaluating model on held-out test images...")
    test_features = extract_features(backbone, splits["test"], batch_size, device, progress)
    with torch.no_grad():
        predictions = head(test_features).argmax(1).numpy()
    metrics = classification_metrics(labels["test"].numpy(), predictions)
    metrics.update(training_seconds=training_seconds, feature_extraction_seconds=extraction_seconds,
                   head_training_seconds=training_seconds - extraction_seconds,
                   train_images=len(splits["train"]), validation_images=len(splits["validation"]),
                   training_image_presentations=len(splits["train"]) * config["epochs"],
                   training_unique_images_per_second=len(splits["train"]) / training_seconds,
                   training_presentations_per_second=len(splits["train"]) * config["epochs"] / training_seconds,
                   best_epoch=min(history, key=lambda r: r["val_loss"])["epoch"], resources=monitor.summary())
    if run_benchmark:
        progress("Warming up and measuring inference latency...")
        metrics.update(benchmark(backbone, head, splits["test"], device))
    return {"metrics": metrics, "history": history, "resource_samples": monitor.samples,
            "head": head, "predictions": predictions.tolist(), "y_true": labels["test"].tolist()}
