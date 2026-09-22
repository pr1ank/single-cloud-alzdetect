"""Assumption-based cloud scenarios, never measurements of cloud hardware."""
from backend.cost_service import estimate_cost


def simulate_aws(metrics, pricing, dataset_bytes, device):
    # Illustrative scenario multipliers, NOT calibrated hardware speedups.
    factor = .25 if device == "cpu" else 1.0
    seconds = metrics["training_seconds"] * factor
    cost = estimate_cost(seconds, pricing["hourly_usd"], dataset_bytes, metrics["train_images"], pricing["storage_gb_month_usd"])
    scenarios = []
    for multiplier in (.5, 1, 2):
        t = seconds * multiplier
        c = estimate_cost(t, pricing["hourly_usd"], dataset_bytes, metrics["train_images"], pricing["storage_gb_month_usd"])
        scenarios.append({"time_factor": factor * multiplier, "training_seconds": t, "total_usd": c["total_usd"]})
    return {"category": "ESTIMATED", "instance": "g4dn.xlarge", "region": "us-east-1", "time_factor": factor,
            "estimated_training_seconds": seconds, "estimated_latency_ms": metrics.get("latency_mean_ms", 0) * factor if "latency_mean_ms" in metrics else None,
            "estimated_train_images_per_second": metrics["train_images"] / seconds,
            "accuracy": None, "cost": cost, "pricing": pricing, "sensitivity": scenarios,
            "assumption": "Local time × 0.25 for CPU or × 1.0 for CUDA; uncalibrated project scenario. Same multiplier for latency. Sensitivity uses half/double this factor. AWS accuracy is not estimated.",
            "disclosure": "Estimated AWS Performance. No AWS infrastructure was actually launched."}
