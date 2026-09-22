import pytest
from backend.efficiency_service import efficiency, resource_score


def example(accuracy=.8, seconds=600, cost=.1, latency=50):
    return efficiency({"accuracy": accuracy, "training_seconds": seconds, "latency_mean_ms": latency,
                       "train_images": 600, "resources": {"cpu_percent": 65, "ram_percent": 50, "gpu_percent": None}},
                      {"cost": {"total_usd": cost}})


def test_known_score_and_units():
    e = example()
    assert e["score"] == pytest.approx(.3 * 80 + .25 * 50 + .2 * 50 + .15 * 50 + .1 * 100)
    assert e["training_efficiency_accuracy_per_second"] == pytest.approx(.8 / 600)
    assert e["cost_efficiency_accuracy_per_usd"] == 8
    assert e["throughput_unique_images_per_second"] == 1


def test_monotonic_and_bounded():
    assert example(seconds=300)["score"] > example()["score"]
    assert example(cost=.05)["score"] > example()["score"]
    for value in example(seconds=1e-6, cost=1e-8, latency=1e-4)["component_scores"].values():
        assert 0 <= value <= 100


def test_saturation_not_rewarded():
    assert resource_score({"cpu_percent": 100, "ram_percent": 100}) < resource_score({"cpu_percent": 65, "ram_percent": 50})
    assert resource_score({"cpu_percent": 0, "ram_percent": 50, "gpu_percent": 65}) == 100


@pytest.mark.parametrize("kwargs", [{"seconds": 0}, {"cost": -1}, {"accuracy": 1.5}, {"latency": float('nan')}])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        example(**kwargs)
