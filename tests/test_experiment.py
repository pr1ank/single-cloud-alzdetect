import pandas as pd
from backend.experiment_service import new_experiment_directory, save_run
from backend.model_service import make_head
from backend.metrics_service import classification_metrics
from backend.utils import read_json
from backend.monitoring_service import automatic_config, ResourceMonitor


def test_experiment_storage(tmp_path):
    a, b = new_experiment_directory(tmp_path), new_experiment_directory(tmp_path)
    assert a != b
    result = {"metrics": classification_metrics([0, 1, 2, 3], [0, 1, 2, 3]),
              "history": [{"epoch": 1, "loss": .4}], "resource_samples": [{"cpu_percent": 20}],
              "head": make_head(), "y_true": [0, 1, 2, 3], "predictions": [0, 1, 2, 3]}
    save_run(a, {"seed": 42}, result)
    for file in ("config.json", "metrics.json", "training_history.csv", "classification_report.csv", "confusion_matrix.csv", "classifier.pt"):
        assert (a / file).is_file()
    assert read_json(a / "metrics.json")["accuracy"] == 1
    assert pd.read_csv(a / "confusion_matrix.csv").shape == (4, 5)


def test_automatic_cpu_and_gpu_configuration():
    strong = automatic_config({"device": "cpu", "ram_gb": 16, "available_ram_gb": 8})
    weak = automatic_config({"device": "cpu", "ram_gb": 4, "available_ram_gb": 2})
    gpu = automatic_config({"device": "cuda", "gpu_memory_gb": 8})
    assert (strong["epochs"], strong["batch_size"]) == (5, 16)
    assert (weak["epochs"], weak["batch_size"]) == (3, 8)
    assert (gpu["epochs"], gpu["batch_size"]) == (10, 32)


def test_monitor_without_gpu():
    with ResourceMonitor() as monitor:
        pass
    report = monitor.summary()
    assert report["sample_count"] >= 2
    assert report["gpu_percent"] is None
    assert report["process_ram_mb"] > 0
