"""Anonymous AWS website pricing with a bundled, explicitly dated fallback."""
import math
from datetime import datetime, timezone
from backend.utils import ROOT, public_json, read_json

PRICING_URL = "https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json"


def parse_ec2_price(payload, instance="g4dn.xlarge"):
    matches = []
    def walk(node):
        if isinstance(node, dict):
            if node.get("Instance Type", node.get("instanceType")) == instance:
                price = float(node["price"])
                if math.isfinite(price) and 0 < price < 100:
                    matches.append(price)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(payload)
    if not matches or len(set(matches)) != 1:
        raise ValueError("AWS response has no unique g4dn.xlarge hourly price")
    return matches[0]


def get_pricing():
    price = read_json(ROOT / "data" / "fallback_pricing.json")
    price["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    price["storage_source"] = "Bundled public pricing reference; EBS gp3"
    try:
        price["hourly_usd"] = parse_ec2_price(public_json(PRICING_URL))
        price["source"] = "Public pricing reference"
        price["compute_source_url"] = PRICING_URL
    except Exception as exc:
        price["fallback_reason"] = f"{type(exc).__name__}: {exc}"
    return price
