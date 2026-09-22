"""Developer verification: run the actual default application and navigate all pages."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest

app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=1800)
app.run()
assert not app.exception, str(app.exception)
assert not app.error, str(app.error)
result = app.session_state["result"]
dataset = app.session_state["dataset"]
print(json.dumps({"dataset": dataset["name"], "demo": dataset["demo"], "pretrained": result["model_info"]["pretrained"],
                  "folder": result["folder"], "metrics": result["metrics"], "scalability": result["scalability"]}, indent=2), flush=True)
for page in ["Overview", "Dataset", "Model Performance", "Cloud Efficiency", "Scalability", "Paper Comparison", "Experiment Details", "Methodology"]:
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, f"{page}: {app.exception}"
    assert not app.error, f"{page}: {app.error}"
    print(f"PAGE OK: {page}", flush=True)
# Verify startup reuses the saved experiment, without pre-populating session state.
second = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
assert not second.exception and not second.error, (str(second.exception), str(second.error), [c.value for c in second.code])
assert second.session_state["result"]["folder"] == result["folder"]
print("DISK CACHE REUSE OK", flush=True)
