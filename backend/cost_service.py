"""Narrow training compute + EBS scenario, not an AWS bill."""
import math


def estimate_cost(training_seconds, hourly_usd, dataset_bytes, images, storage_gb_month_usd=.08):
    values = (training_seconds, hourly_usd, dataset_bytes, images, storage_gb_month_usd)
    if not all(math.isfinite(float(x)) for x in values) or min(values) < 0 or images <= 0:
        raise ValueError("Cost inputs must be finite and nonnegative; images must be positive")
    billed_seconds = max(60, math.ceil(training_seconds))
    # A small boot/data volume, retained only for the modeled training window.
    volume_gb = max(30, math.ceil(dataset_bytes / 2**30) + 10)
    compute = billed_seconds / 3600 * hourly_usd
    storage = volume_gb * storage_gb_month_usd * billed_seconds / (30 * 86400)
    return {"compute_usd": compute, "storage_usd": storage, "total_usd": compute + storage,
            "cost_per_image_usd": (compute + storage) / images, "billed_seconds": billed_seconds,
            "volume_gb": volume_gb, "cost_denominator_images": images,
            "scope": "One training run; compute plus gp3 storage retained during run; 60-second minimum; excludes setup, downloads, evaluation, idle, egress, tax, snapshots and long-term storage"}
