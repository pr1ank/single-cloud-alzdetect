import json
import shutil
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from backend.dataset_service import (CLASSES, SOURCES, acquire_dataset, canonical_class, discover_images,
                                     extract_parquet, generate_demo, split_dataset, stratified_subset, validate_dataset)


def test_nested_class_discovery_and_validation(tmp_path):
    root = generate_demo(tmp_path / "arbitrary" / "train")
    assert canonical_class("Very_Mild_Demented") == "VeryMildDemented"
    assert len(discover_images(tmp_path)) == 80
    (root / CLASSES[0] / "broken.jpg").write_bytes(b"not an image")
    source = root / CLASSES[0] / "synthetic_000.png"
    shutil.copyfile(source, root / CLASSES[0] / "duplicate.png")
    report = validate_dataset(tmp_path)
    assert report["corrupted_count"] == 1
    assert report["duplicates_removed"] == 1
    assert report["valid_images"] == 80
    assert report["class_count"] == 4


def test_conflicting_duplicates_excluded(tmp_path):
    generate_demo(tmp_path)
    shutil.copyfile(tmp_path / CLASSES[0] / "synthetic_000.png", tmp_path / CLASSES[1] / "conflict.png")
    result = validate_dataset(tmp_path)
    assert result["conflicting_hashes_removed"] == 1
    assert result["valid_images"] == 79


def test_stratified_deterministic_disjoint_splits(tmp_path):
    rows = validate_dataset(generate_demo(tmp_path))["rows"]
    split = split_dataset(rows)
    assert split == split_dataset(rows)
    assert {k: len(v) for k, v in split.items()} == {"train": 56, "validation": 12, "test": 12}
    sets = [{r["sha256"] for r in split[name]} for name in split]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    small = stratified_subset(split["train"], .1)
    large = stratified_subset(split["train"], .5)
    assert {r["path"] for r in small} <= {r["path"] for r in large}
    assert len({r["class"] for r in small}) == 4


def test_both_download_failures_use_real_demo_fallback(tmp_path, monkeypatch):
    attempts = []
    def fail(url, target):
        attempts.append(url)
        raise ConnectionError("Simulated network outage")
    monkeypatch.setattr("backend.dataset_service.download", fail)
    dataset = acquire_dataset(tmp_path)
    assert len(attempts) == 2
    assert dataset["demo"] is True
    assert dataset["valid_images"] == 80
    assert len(dataset["download_errors"]) == 2
    assert acquire_dataset(tmp_path)["fingerprint"] == dataset["fingerprint"]
    assert len(attempts) == 2


def test_parquet_metadata_controls_label_mapping(tmp_path):
    root = generate_demo(tmp_path / "samples")
    images = [{"bytes": (root / c / "synthetic_000.png").read_bytes(), "path": None} for c in reversed(CLASSES)]
    table = pa.table({"image": images, "label": [0, 1, 2, 3]})
    metadata = {"info": {"features": {"label": {"names": list(reversed(CLASSES))}}}}
    table = table.replace_schema_metadata({b"huggingface": json.dumps(metadata).encode()})
    file = tmp_path / "test.parquet"
    pq.write_table(table, file)
    extract_parquet(file, tmp_path / "extracted")
    actual = sorted(r["class"] for r in discover_images(tmp_path / "extracted"))
    assert actual == sorted(CLASSES)
    assert (tmp_path / "extracted" / CLASSES[-1] / "test_000000.jpg").read_bytes() == images[0]["bytes"]
    # The verified primary source omits Arrow metadata; use its pinned card labels.
    pq.write_table(table.replace_schema_metadata(None), file)
    extract_parquet(file, tmp_path / "without_metadata", list(reversed(CLASSES)))
    assert len(discover_images(tmp_path / "without_metadata")) == 4


def test_secondary_source_automatic_success(tmp_path, monkeypatch):
    demo = generate_demo(tmp_path / "source")
    calls = []
    def download(url, target):
        calls.append(url)
        if "Falah" in url:
            raise ConnectionError("primary unavailable")
        return target
    def extract(path, target, source_label_names=None):
        shutil.copytree(demo, target, dirs_exist_ok=True)
    monkeypatch.setattr("backend.dataset_service.download", download)
    monkeypatch.setattr("backend.dataset_service.extract_parquet", extract)
    result = acquire_dataset(tmp_path)
    assert not result["demo"]
    assert result["name"] == SOURCES[1]["repo"]
    assert len(calls) == 4
