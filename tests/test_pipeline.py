"""Real tiny CPU training + all dashboard pages, without network or fake experiment metrics."""
import json
from pathlib import Path
import pytest
import torch
from torch import nn
from torchvision.models import efficientnet_b0
from streamlit.testing.v1 import AppTest
from backend.dataset_service import acquire_dataset
from backend.experiment_service import run_automatic
from backend.model_service import seed_everything
from backend.monitoring_service import detect_hardware, automatic_config
from backend.utils import ROOT, read_json


@pytest.fixture(scope="module")
def measured_demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("real_demo_pipeline")
    dataset = acquire_dataset(root, sources=[])
    hardware = detect_hardware()
    hardware.update(device="cpu", gpu=None)
    config = automatic_config(hardware)
    config.update(epochs=1, max_images=40, batch_size=8, scalability_epochs=1)
    def loader(device):
        seed_everything()
        model = efficientnet_b0(weights=None)
        model.classifier = nn.Identity()
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, {"pretrained": False, "warning": "Integration test: random frozen features on synthetic DEMO DATA", "model": "EfficientNetB0"}
    result = run_automatic(dataset, hardware, config, read_json(ROOT / "data" / "fallback_pricing.json"), loader, root=root)
    return dataset, result, root, hardware, config


def test_real_training_and_scalability(measured_demo):
    dataset, result, root, _, _ = measured_demo
    assert dataset["demo"]
    assert result["metrics"]["training_seconds"] > 0
    assert 0 <= result["metrics"]["accuracy"] <= 1
    assert result["metrics"]["latency_mean_ms"] > 0
    assert result["cloud"]["cost"]["total_usd"] > 0
    assert [r["status"] for r in result["scalability"]] == ["measured"] * 5
    assert len(list(root.glob("experiments/*/scalability/*/metrics.json"))) == 5


def test_completed_run_reused_without_training(measured_demo):
    dataset, result, root, hardware, config = measured_demo
    def forbidden(device):
        raise AssertionError("Completed cached run must not reload/train model")
    reused = run_automatic(dataset, hardware, config, read_json(ROOT / "data" / "fallback_pricing.json"), forbidden, root=root)
    assert reused["folder"] == result["folder"]


@pytest.mark.parametrize("page", ["Overview", "Dataset", "Model Performance", "Cloud Efficiency", "Scalability", "Paper Comparison", "Experiment Details", "Methodology"])
def test_all_pages_render_measured_results(measured_demo, page):
    dataset, result, *_ = measured_demo
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app.session_state["dataset"] = dataset
    app.session_state["result"] = result
    app.run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, str(app.exception)
    assert not app.error, str(app.error)
    assert any("DEMO DATA" in w.value for w in app.warning)
    expected_charts = {"Overview": 1, "Dataset": 1, "Model Performance": 3, "Cloud Efficiency": 1,
                       "Scalability": 4, "Paper Comparison": 3}
    if page in expected_charts:
        assert len(app.get("plotly_chart")) == expected_charts[page]


def test_failed_run_shows_unavailable():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app.session_state["experiment_error"] = "Deliberate failure for error-path test"
    app.run()
    assert not app.exception
    assert app.error[0].value == "Experiment unavailable"


def test_multiple_automatic_sessions_reuse_cached_dataset(measured_demo, monkeypatch):
    dataset, result, *_ = measured_demo
    def acquisition(*args, progress=lambda message: None, **kwargs):
        progress("Dataset ready.")
        return dataset
    monkeypatch.setattr("backend.dataset_service.acquire_dataset", acquisition)
    monkeypatch.setattr("backend.pricing_service.get_pricing", lambda: result["cloud"]["pricing"])
    monkeypatch.setattr("backend.experiment_service.run_automatic", lambda *args, **kwargs: result)
    for _ in range(2):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        assert not app.exception, str(app.exception)
        assert not app.error, [c.value for c in app.code]
