"""Public Parquet acquisition, folder discovery, validation, and deterministic splits."""
from __future__ import annotations
import hashlib
import io
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from filelock import FileLock

from backend.utils import ROOT, download, fingerprint, read_json, write_json

CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
SOURCES = [
    {"repo": "Falah/Alzheimer_MRI", "revision": "daac24f9597236b45837d82f7eb9c9ad1f8c60c8",
     "label_names": ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"],
     "license": "apache-2.0 (publisher metadata)", "files": [
         "data/train-00000-of-00001-c08a401c53fe5312.parquet",
         "data/test-00000-of-00001-44110b9df98c5585.parquet"]},
    {"repo": "Obscure-Entropy/Alzheimer-MRI", "revision": "02dc8a193fc452348e45d9192526076f1b11a89c",
     "label_names": ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"],
     "license": "mit (publisher metadata)", "files": [
         "data/train-00000-of-00001-563eb85ed689e7ad.parquet",
         "data/valid-00000-of-00001-65ca90615ebcd406.parquet",
         "data/test-00000-of-00001-d3a8e2b8f3f69343.parquet"]},
]


def canonical_class(name):
    normalized = re.sub(r"[^a-z]", "", str(name).lower())
    return next((c for c in CLASSES if c.lower() == normalized), None)


def discover_images(root: Path):
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        label = next((canonical_class(p) for p in reversed(path.relative_to(root).parts[:-1])
                      if canonical_class(p)), None)
        if label:
            rows.append({"path": str(path.resolve()), "class": label})
    return rows


def validate_dataset(root: Path):
    found = discover_images(root)
    valid, corrupt, duplicates, conflicts = [], [], [], set()
    hashes = {}
    dimensions = Counter()
    for row in found:
        try:
            with Image.open(row["path"]) as im:
                im.load()
                rgb = im.convert("RGB")
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                size = f"{im.width} x {im.height}"
                if im.width < 8 or im.height < 8:
                    raise ValueError("Image is too small")
            if digest in hashes:
                duplicates.append(row["path"])
                if hashes[digest] != row["class"]:
                    conflicts.add(digest)
                continue
            hashes[digest] = row["class"]
            valid.append({**row, "sha256": digest, "dimensions": size})
            dimensions[size] += 1
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            corrupt.append({"path": row["path"], "error": str(exc)})
    valid = [r for r in valid if r["sha256"] not in conflicts]
    counts = dict(Counter(r["class"] for r in valid))
    return {"rows": valid, "total_found": len(found), "valid_images": len(valid),
            "class_count": len(counts), "class_distribution": counts,
            "corrupted_count": len(corrupt), "corrupted": corrupt,
            "duplicates_removed": len(duplicates), "conflicting_hashes_removed": len(conflicts),
            "dimensions": dict(dimensions),
            "size_bytes": sum(Path(r["path"]).stat().st_size for r in valid),
            "fingerprint": fingerprint([(r["sha256"], r["class"]) for r in valid])}


def usable(report):
    return report["class_count"] == 4 and min(report["class_distribution"].values()) >= 10


def generate_demo(root: Path, per_class=20):
    """Procedural geometric noise images, deliberately not MRI or disease evidence."""
    rng = np.random.default_rng(42)
    for label in CLASSES:
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            path = folder / f"synthetic_{i:03d}.png"
            if path.exists():
                continue
            im = Image.fromarray(rng.integers(0, 70, (96, 96), dtype=np.uint8))
            draw = ImageDraw.Draw(im)
            x, y = (int(v) for v in rng.integers(8, 35, 2))
            draw.rectangle((x, y, x + 38, y + 38), outline=180, width=3)
            draw.text((5, 80), "SYNTHETIC", fill=255)
            im.save(path)
    return root


def extract_parquet(path: Path, target: Path, source_label_names=None):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    metadata = json.loads((parquet.schema_arrow.metadata or {}).get(b"huggingface", b"{}"))
    # Some exports drop Arrow metadata. Each pinned source's dataset card supplies
    # its verified label order; never infer numeric class mappings from folder order.
    names = metadata.get("info", {}).get("features", {}).get("label", {}).get("names") or source_label_names
    if not names:
        raise ValueError("Dataset lacks explicit class-label metadata")
    if isinstance(names, dict):
        names = [names[str(i)] for i in range(len(names))]
    labels = [canonical_class(n) for n in names]
    if set(labels) != set(CLASSES):
        raise ValueError(f"Unexpected dataset class labels: {names}")
    index = 0
    for batch in parquet.iter_batches(batch_size=128, columns=["image", "label"]):
        for row in batch.to_pylist():
            label = labels[int(row["label"])]
            folder = target / label
            folder.mkdir(parents=True, exist_ok=True)
            blob = row["image"].get("bytes")
            if not blob:
                raise ValueError("Expected embedded public image bytes")
            # Preserve original bytes, including corrupt records for validation accounting.
            (folder / f"{path.stem}_{index:06d}.jpg").write_bytes(blob)
            index += 1


def acquire_dataset(root=ROOT, progress=lambda message: None, sources=None):
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    with FileLock(str(data / "acquisition.lock"), timeout=600):
        manifest_path = data / "dataset_manifest.json"
        if manifest_path.exists():
            try:
                manifest = read_json(manifest_path)
                if not manifest["demo"] or time.time() - manifest["acquired_at"] < 86400:
                    report = validate_dataset(Path(manifest["root"]))
                    if usable(report):
                        progress("Dataset ready — validated local cache.")
                        return {**manifest, **report}
            except (OSError, ValueError, KeyError):
                pass
        errors = []
        for source in SOURCES if sources is None else sources:
            try:
                progress(f"Downloading public dataset: {source['repo']}")
                destination = data / "public" / source["repo"].split("/")[0] / source["revision"]
                for name in source["files"]:
                    cached = data / "downloads" / source["repo"].split("/")[0] / Path(name).name
                    if not cached.exists():
                        url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{name}"
                        download(url, cached)
                    progress(f"Extracting {Path(name).name}")
                    extract_parquet(cached, destination, source["label_names"])
                report = validate_dataset(destination)
                if not usable(report):
                    raise ValueError("Dataset validation requires four classes with at least 10 unique images each")
                manifest = {"name": source["repo"], "source": f"https://huggingface.co/datasets/{source['repo']}",
                            "revision": source["revision"], "license": source["license"], "demo": False,
                            "root": str(destination.resolve()), "acquired_at": time.time(), "download_errors": errors}
                write_json(manifest_path, manifest)
                progress("Dataset ready.")
                return {**manifest, **report}
            except Exception as exc:
                errors.append(f"{source['repo']}: {type(exc).__name__}: {exc}")
                progress("Public source unavailable; trying the next automatic fallback.")
        destination = generate_demo(data / "demo")
        report = validate_dataset(destination)
        manifest = {"name": "DEMO DATA — synthetic geometric noise", "source": "Bundled procedural generator (seed 42)",
                    "revision": "synthetic-v1", "license": "Project-generated synthetic data", "demo": True,
                    "root": str(destination.resolve()), "acquired_at": time.time(), "download_errors": errors}
        write_json(manifest_path, manifest)
        progress("Demo dataset is being used because the public dataset could not be retrieved.")
        return {**manifest, **report}


def stratified_subset(rows, fraction=1.0, max_images=None, minimum_per_class=1):
    rng = np.random.default_rng(42)
    effective = min(fraction, max_images / len(rows)) if max_images else fraction
    selected = []
    for label in CLASSES:
        group = [r for r in rows if r["class"] == label]
        order = rng.permutation(len(group))
        n = min(len(group), max(minimum_per_class, int(round(len(group) * effective))))
        selected.extend(group[i] for i in order[:n])
    return selected


def split_dataset(rows):
    """Per-class 70/15/15 allocation, with rounding and at least one item per split."""
    splits = {"train": [], "validation": [], "test": []}
    rng = np.random.default_rng(42)
    for label in CLASSES:
        group = [r for r in rows if r["class"] == label]
        if len(group) < 3:
            raise ValueError(f"Not enough unique images for class {label}")
        group = [group[i] for i in rng.permutation(len(group))]
        n_train = min(len(group) - 2, max(1, int(len(group) * .70)))
        n_val = min(len(group) - n_train - 1, max(1, int(round(len(group) * .15))))
        splits["train"].extend(group[:n_train])
        splits["validation"].extend(group[n_train:n_train + n_val])
        splits["test"].extend(group[n_train + n_val:])
    return splits
