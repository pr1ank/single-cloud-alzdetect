import pytest
from backend.cost_service import estimate_cost
from backend.pricing_service import parse_ec2_price, get_pricing
from backend.cloud_simulation_service import simulate_aws


def test_known_cost():
    result = estimate_cost(3600, .526, 1024, 1000)
    assert result["compute_usd"] == .526
    assert result["storage_usd"] == pytest.approx(30 * .08 / 720)
    assert result["cost_per_image_usd"] == result["total_usd"] / 1000


def test_minimum_billing_and_invalid():
    assert estimate_cost(1, .526, 0, 1)["billed_seconds"] == 60
    for value in (-1, float('inf'), float('nan')):
        with pytest.raises(ValueError):
            estimate_cost(value, .526, 0, 1)
    with pytest.raises(ValueError):
        estimate_cost(60, .526, 0, 0)


def test_pricing_parser():
    payload = {"regions": {"US East (N. Virginia)": {"a": {"Instance Type": "g4dn.xlarge", "price": "0.5260000000"}}}}
    assert parse_ec2_price(payload) == .526
    with pytest.raises(ValueError):
        parse_ec2_price({})
    with pytest.raises(ValueError):
        parse_ec2_price([{"Instance Type": "g4dn.xlarge", "price": ".526"}, {"Instance Type": "g4dn.xlarge", "price": ".7"}])


def test_pricing_fallback_and_live_parsing(monkeypatch):
    def unavailable(url):
        raise ConnectionError("Offline test")
    monkeypatch.setattr("backend.pricing_service.public_json", unavailable)
    assert get_pricing()["source"] == "Fallback pricing reference"
    monkeypatch.setattr("backend.pricing_service.public_json", lambda url: {"Instance Type": "g4dn.xlarge", "price": ".6"})
    result = get_pricing()
    assert result["hourly_usd"] == .6
    assert result["source"] == "Public pricing reference"


def test_cloud_scenario_does_not_fabricate_accuracy():
    result = simulate_aws({"training_seconds": 100, "train_images": 100, "latency_mean_ms": 40},
                          {"hourly_usd": .526, "storage_gb_month_usd": .08}, 10000, "cpu")
    assert result["estimated_training_seconds"] == 25
    assert result["estimated_latency_ms"] == 10
    assert result["accuracy"] is None
    assert len(result["sensitivity"]) == 3
