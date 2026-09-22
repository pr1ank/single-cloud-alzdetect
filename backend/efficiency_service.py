"""Transparent, fixed-anchor project scores; not clinical or industry standards."""
import math


def resource_score(resources):
    cpu = resources.get("cpu_percent")
    ram = resources.get("ram_percent")
    gpu = resources.get("gpu_percent")
    if cpu is None or ram is None:
        raise ValueError("Measured CPU and RAM are required")
    # Broad utilization target; both idle and saturated compute reduce this proxy.
    compute = gpu if gpu is not None else cpu
    compute_score = max(0, 100 * (1 - abs(compute - 65) / 65))
    memory_score = max(0, 100 * (1 - max(0, ram - 70) / 30))
    return .6 * compute_score + .4 * memory_score


def efficiency(metrics, cloud):
    accuracy, seconds = metrics["accuracy"], metrics["training_seconds"]
    cost, latency = cloud["cost"]["total_usd"], metrics["latency_mean_ms"]
    if not 0 <= accuracy <= 1 or not all(math.isfinite(x) and x > 0 for x in (seconds, cost, latency)):
        raise ValueError("Efficiency requires valid measured accuracy, time, latency and positive estimated cost")
    scores = {"accuracy": 100 * accuracy, "training_time": 100 / (1 + seconds / 600),
              "cost": 100 / (1 + cost / .10), "latency": 100 / (1 + latency / 50),
              "resource": resource_score(metrics["resources"])}
    weights = {"accuracy": .30, "training_time": .25, "cost": .20, "latency": .15, "resource": .10}
    return {"training_efficiency_accuracy_per_second": accuracy / seconds,
            "cost_efficiency_accuracy_per_usd": accuracy / cost,
            "throughput_unique_images_per_second": metrics["train_images"] / seconds,
            "resource_efficiency": scores["resource"], "component_scores": scores,
            "score": sum(weights[key] * value for key, value in scores.items()),
            "note": "This is a project-defined composite efficiency score and is not an industry-standard metric."}
