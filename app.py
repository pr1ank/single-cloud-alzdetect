"""Run with: streamlit run app.py. No configuration, credentials or uploads."""
from __future__ import annotations
import importlib
import json
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Single-Cloud Alzheimer's Analysis", page_icon="🧠", layout="wide")


def check_dependencies():
    missing = []
    for name in ("torch", "torchvision", "numpy", "pandas", "PIL", "sklearn", "psutil", "requests", "pyarrow", "plotly", "filelock"):
        try:
            importlib.import_module(name)
        except Exception as exc:
            missing.append(f"{name}: {type(exc).__name__}: {exc}")
    if missing:
        st.error("Required packages are missing or incompatible. Install the project's requirements in the Python environment used to start Streamlit.")
        st.code("pip install -r requirements.txt")
        st.code("\n".join(missing))
        st.stop()


check_dependencies()
import pandas as pd
import plotly.express as px
from backend.utils import ROOT
from backend.dataset_service import acquire_dataset, CLASSES
from backend.model_service import load_backbone
from backend.monitoring_service import detect_hardware, automatic_config
from backend.pricing_service import get_pricing
from backend.experiment_service import run_automatic
from backend.metrics_service import report_frame

st.markdown("""<style>
.stApp { background: #f6f8fc; color: #17233b; }
[data-testid="stSidebar"] { background: #eaf0f8; }
[data-testid="stMetric"] { background: white; border: 1px solid #dde5f0; border-radius: 14px; padding: 16px; }
[data-testid="stMetricLabel"] { color: #536782; }
h1, h2, h3 { letter-spacing: -0.03em; }
.eyebrow {color:#147d92; font-weight:700; letter-spacing:.15em; font-size:.75rem; margin-top:12px;}
.hero {font-size:1.15rem; color:#61738b; margin-bottom:1.4rem;}
.block-container {padding-top:2rem; max-width:1450px;}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_dataset():
    # This element must belong to the cached function so Streamlit can replay it
    # in another session without referencing a previous session's layout block.
    acquisition_status = st.empty()
    return acquire_dataset(progress=acquisition_status.write)


@st.cache_resource(ttl=3600, show_spinner=False)
def cached_model(device):
    return load_backbone(device)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_pricing():
    return get_pricing()


@st.cache_data(ttl=300, show_spinner=False)
def cached_hardware():
    return detect_hardware()


def cards(items):
    for start in range(0, len(items), 4):
        columns = st.columns(min(4, len(items) - start))
        for col, (label, value) in zip(columns, items[start:start + 4]):
            col.metric(label, value)


def chart(fig):
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", font_color="#17233b", margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


def percentage(value):
    return f"{value:.1%}"


st.markdown('<div class="eyebrow">SINGLE-CLOUD RESEARCH PLATFORM</div>', unsafe_allow_html=True)
st.title("Single-Cloud Alzheimer's Analysis")
st.markdown('<div class="hero">Performance, Cost and Resource Efficiency Evaluation</div>', unsafe_allow_html=True)
st.caption("Research and educational use only. This system is not a medical diagnostic device.")

hardware = cached_hardware()
with st.expander("Detected Hardware", expanded="result" not in st.session_state):
    st.write(f"**CPU:** {hardware['cpu']} · {hardware['cpu_count']} logical cores")
    st.write(f"**RAM:** {hardware['ram_gb']:.1f} GiB total · {hardware['available_ram_gb']:.1f} GiB available at detection")
    st.write(f"**GPU:** {hardware['gpu']}" if hardware["gpu"] else "GPU not available — CPU execution")
    st.write(f"**Execution Mode:** {'GPU (CUDA backbone / CPU head)' if hardware['device'] == 'cuda' else 'CPU'}")

pages = ["Overview", "Dataset", "Model Performance", "Cloud Efficiency", "Scalability", "Paper Comparison", "Experiment Details", "Methodology"]
st.sidebar.markdown("### Research workspace")
page = st.sidebar.radio("Navigate", pages, label_visibility="collapsed")
st.sidebar.caption("ACTUAL · measured on this machine\n\nESTIMATED · AWS scenario\n\nREFERENCE · supplied paper values")
st.sidebar.divider()
st.sidebar.caption("No AWS resources launched. No credentials required.")

if "result" not in st.session_state and "experiment_error" not in st.session_state:
    with st.status("Initializing...", expanded=True) as status:
        step = st.empty()
        epoch_bar = st.progress(0, text="Preparing automatic experiment")
        live = st.empty()
        config = automatic_config(hardware)
        st.write("Automatic resource-aware configuration")
        st.caption(f"224×224 · batch {config['batch_size']} · {config['epochs']} epochs · requested {config['dataset_fraction']:.0%} dataset · cap {config['max_images']:,} images")
        def progress(message):
            step.write(message)
        def epoch_progress(row):
            epoch_bar.progress(row["epoch"] / config["epochs"], text=f"Epoch {row['epoch']}/{config['epochs']}")
            live.dataframe(pd.DataFrame([row]), hide_index=True, use_container_width=True)
        try:
            progress("Checking dataset...")
            dataset = cached_dataset()
            st.session_state["dataset"] = dataset
            if dataset["demo"]:
                st.warning("Demo dataset is being used because the public dataset could not be retrieved. DEMO DATA: synthetic images, not MRI.")
            progress("Retrieving public AWS pricing...")
            pricing = cached_pricing()
            result = run_automatic(dataset, hardware, config, pricing, cached_model, progress, epoch_progress)
            st.session_state["result"] = result
            epoch_bar.progress(1.0, text="Complete")
            status.update(label="Complete — experiment and scalability results ready", state="complete", expanded=False)
        except Exception as exc:
            st.session_state["experiment_error"] = f"{type(exc).__name__}: {exc}"
            status.update(label="Experiment unavailable", state="error", expanded=True)

dataset = st.session_state.get("dataset")
result = st.session_state.get("result")
if dataset and dataset["demo"]:
    st.warning("DEMO DATA — Demo dataset is being used because the public dataset could not be retrieved. Synthetic geometric images are not Alzheimer's MRI data. Metrics only demonstrate the pipeline.")
if result and not result["model_info"]["pretrained"]:
    st.warning(result["model_info"]["warning"])
if "experiment_error" in st.session_state:
    st.error("Experiment unavailable")
    st.code(st.session_state["experiment_error"])
    if st.button("Retry automatic initialization"):
        st.session_state.pop("experiment_error", None)
        st.rerun()

if page == "Methodology":
    st.markdown((ROOT / "assets" / "METHODOLOGY.md").read_text(encoding="utf-8"))
elif page == "Dataset" and dataset:
    st.header("Dataset")
    st.write(f"**{dataset['name']}**")
    st.write("Source:", dataset["source"])
    st.caption(f"{dataset['license']} · pinned revision {dataset['revision']}")
    cards([("Discovered images", f"{dataset['total_found']:,}"), ("Validated unique images", f"{dataset['valid_images']:,}"),
           ("Classes", str(dataset["class_count"])), ("Validated data size", f"{dataset['size_bytes'] / 2**20:.1f} MiB"),
           ("Corrupt images excluded", str(dataset['corrupted_count'])), ("Exact duplicates excluded", str(dataset['duplicates_removed']))])
    if result:
        split_counts = result["config"]["split_counts"]
        cards([(f"{name.title()} images", str(n)) for name, n in split_counts.items()])
        st.caption(f"Selected {result['config']['selected_images']:,} images ({result['config']['actual_dataset_fraction']:.1%} of validated data). Target split 70/15/15, rounded per class; seed 42.")
    distribution = pd.DataFrame(dataset["class_distribution"].items(), columns=["Class", "Images"])
    chart(px.bar(distribution, x="Class", y="Images", color="Class", title="Validated class distribution", color_discrete_sequence=px.colors.qualitative.Safe))
    st.subheader("Sample images" + (" — SYNTHETIC DEMO" if dataset["demo"] else " — public MRI dataset"))
    for col, label in zip(st.columns(4), CLASSES):
        sample = next(r for r in dataset["rows"] if r["class"] == label)
        col.image(sample["path"], caption=label, use_container_width=True)
    with st.expander("Validation details and source fallback log"):
        st.json({k: dataset[k] for k in ("dimensions", "conflicting_hashes_removed", "corrupted", "download_errors", "fingerprint")})
elif result:
    m, cloud, eff, cfg = result["metrics"], result["cloud"], result["efficiency"], result["config"]
    demo_suffix = " · DEMO" if result["dataset_demo"] or not result["model_info"]["pretrained"] else ""
    if page == "Overview":
        st.caption(f"ACTUAL LOCAL RESULT{demo_suffix} · saved {result['created_at']} · {cfg['device'].upper()} execution")
        cards([("Accuracy", percentage(m["accuracy"])), ("Macro F1", f"{m['f1_macro']:.3f}"),
               ("Training time", f"{m['training_seconds'] / 60:.2f} min"), ("Mean inference latency", f"{m['latency_mean_ms']:.2f} ms"),
               ("Estimated AWS training cost", f"${cloud['cost']['total_usd']:.5f}"), ("Project efficiency score", f"{eff['score']:.1f} / 100"),
               ("Model", "EfficientNetB0"), ("Execution mode", cfg["device"].upper())])
        st.info(f"Dataset: {dataset['name']} · {cfg['selected_images']:,} selected images · {cfg['epochs']} head-training epochs. Automatic resource-aware configuration.")
        a, b = st.columns([3, 2])
        with a:
            st.subheader("Learning progress")
            chart(px.line(pd.DataFrame(result["history"]), x="epoch", y=["accuracy", "val_accuracy"], markers=True, labels={"value": "Accuracy", "variable": "Split"}))
        with b:
            st.subheader("What these results mean")
            st.write("Local metrics come from the completed experiment. AWS cost and performance use explicit scenario assumptions. Paper figures are a separate reference.")
            st.write("The experiment studies the efficiency of one image-classification workload; it does not establish that single-cloud is universally better than multi-cloud.")
            st.caption(eff["note"])
        st.warning("AWS values are estimates. No AWS infrastructure was actually launched.")
    elif page == "Model Performance":
        st.header("Model performance" + demo_suffix)
        st.caption("ACTUAL — held-out test set; best validation-loss classifier")
        cards([("Accuracy", percentage(m["accuracy"])), ("Macro precision", f"{m['precision_macro']:.3f}"),
               ("Macro recall", f"{m['recall_macro']:.3f}"), ("Macro F1", f"{m['f1_macro']:.3f}"),
               ("Weighted precision", f"{m['precision_weighted']:.3f}"), ("Weighted recall", f"{m['recall_weighted']:.3f}"),
               ("Weighted F1", f"{m['f1_weighted']:.3f}"), ("Test images", str(m["test_images"]))])
        chart(px.imshow(m["confusion_matrix"], x=CLASSES, y=CLASSES, text_auto=True, color_continuous_scale="Blues",
                        labels={"x": "Predicted class", "y": "Actual class", "color": "Images"}, title="Confusion matrix"))
        st.dataframe(report_frame(m), use_container_width=True)
        for names, title in [(["accuracy", "val_accuracy"], "Training and validation accuracy"), (["loss", "val_loss"], "Training and validation loss")]:
            chart(px.line(pd.DataFrame(result["history"]), x="epoch", y=names, markers=True, title=title))
        st.subheader("Measured inference benchmark")
        cards([("Mean latency", f"{m['latency_mean_ms']:.2f} ms"), ("P50 / median", f"{m['latency_median_ms']:.2f} ms"),
               ("P95 latency", f"{m['latency_p95_ms']:.2f} ms"), ("Inference images / sec", f"{m['inference_images_per_second']:.2f}")])
        st.caption(m["benchmark_scope"] + f" · {m['warmup_images']} warm-ups; {m['benchmark_images']} measurements.")
    elif page == "Cloud Efficiency":
        st.header("Cloud efficiency")
        st.subheader("Actual local performance" + demo_suffix)
        r = m["resources"]
        cards([("Training time", f"{m['training_seconds']:.2f} s"), ("Mean system CPU", f"{r['cpu_percent']:.1f}%"),
               ("Mean system RAM", f"{r['ram_percent']:.1f}%"), ("Peak process RAM", f"{r['process_ram_mb_peak']:.0f} MiB"),
               ("Mean latency", f"{m['latency_mean_ms']:.2f} ms"), ("Training images / sec", f"{m['training_unique_images_per_second']:.2f}")])
        if cfg["device"] == "cpu":
            st.info("GPU not available — CPU execution" if hardware["gpu"] is None else "CPU execution after automatic memory recovery")
        else:
            st.write(f"GPU utilization: {r['gpu_percent']:.1f}%" if r["gpu_percent"] is not None else "GPU utilization counter unavailable")
            st.write(f"GPU memory: {r['gpu_memory_mb']:.0f} MiB" if r["gpu_memory_mb"] is not None else "GPU memory counter unavailable")
        st.caption(r["scope"])
        st.subheader("Estimated AWS Performance")
        st.write("**g4dn.xlarge · Linux · us-east-1 · On-Demand**")
        cards([("Estimated training time", f"{cloud['estimated_training_seconds']:.2f} s"), ("Estimated compute cost", f"${cloud['cost']['compute_usd']:.5f}"),
               ("Estimated storage cost", f"${cloud['cost']['storage_usd']:.6f}"), ("Estimated total cost", f"${cloud['cost']['total_usd']:.5f}"),
               ("Estimated cost / training image", f"${cloud['cost']['cost_per_image_usd']:.8f}"), ("Estimated training images / sec", f"{cloud['estimated_train_images_per_second']:.2f}")])
        st.caption(f"{cloud['pricing']['source']} · ${cloud['pricing']['hourly_usd']:.3f}/hour · pricing date {cloud['pricing'].get('retrieved_at', cloud['pricing']['reference_date'])}")
        st.write(cloud["assumption"])
        st.dataframe(pd.DataFrame(cloud["sensitivity"]), hide_index=True, use_container_width=True)
        st.caption(cloud["cost"]["scope"])
        st.warning("AWS values are estimates based on public pricing and project-defined performance scaling assumptions. No AWS resources were launched.")
        st.subheader("Efficiency")
        cards([("Accuracy / training second", f"{eff['training_efficiency_accuracy_per_second']:.6f}"),
               ("Accuracy / estimated USD", f"{eff['cost_efficiency_accuracy_per_usd']:.2f}"),
               ("Resource efficiency", f"{eff['resource_efficiency']:.1f} / 100"), ("Project efficiency score", f"{eff['score']:.1f} / 100")])
        chart(px.bar(pd.DataFrame(eff["component_scores"].items(), columns=["Component", "Score"]), x="Component", y="Score", range_y=[0, 100], title="Normalized component scores"))
        st.caption(eff["note"])
    elif page == "Scalability":
        st.header("Scalability experiments" + demo_suffix)
        st.info(f"10–100% of the selected training pool ({cfg['split_counts']['train']} images), not of the full public dataset. Fresh head and feature extraction per run; fixed validation/test sets; {cfg['scalability_epochs']} epoch(s) per size to control runtime.")
        frame = pd.DataFrame(result["scalability"])
        st.dataframe(frame, use_container_width=True, hide_index=True)
        valid = frame[frame["status"] == "measured"]
        if not valid.empty:
            for y, label in [("training_seconds", "Measured training time (seconds)"), ("accuracy", "Measured test accuracy"),
                             ("estimated_aws_usd", "Estimated AWS training cost (USD)"), ("throughput_images_s", "Measured training throughput (images/sec)")]:
                chart(px.line(valid, x="train_images", y=y, markers=True, title=label, labels={"train_images": "Unique training images", y: label}))
        st.caption("Single runs without confidence intervals. Accuracy and time need not be monotonic; these reduced-epoch results are not equivalent to the main training run.")
    elif page == "Paper Comparison":
        st.header("Paper comparison")
        reference = pd.read_csv(ROOT / "data" / "reference_paper_results.csv")
        ours = pd.DataFrame([
            {"source": "Our Experiment" + demo_suffix, "category": "ACTUAL", "architecture": "Local single-machine workload", "accuracy_percent": m["accuracy"] * 100, "training_minutes": m["training_seconds"] / 60, "latency_ms": m["latency_mean_ms"]},
            {"source": "Our Estimated AWS" + demo_suffix, "category": "ESTIMATED", "architecture": "Single-cloud AWS scenario", "accuracy_percent": None, "training_minutes": cloud["estimated_training_seconds"] / 60, "latency_ms": cloud["estimated_latency_ms"]},
        ])
        frame = pd.concat([ours, reference], ignore_index=True)
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.caption("AWS accuracy is unavailable; no cloud model was trained. REFERENCE rows are user-supplied paper figures, not independently verified against a paper file.")
        for y, title in [("accuracy_percent", "Accuracy (%)"), ("training_minutes", "Training time (minutes)"), ("latency_ms", "Inference latency (ms)")]:
            chart(px.bar(frame.dropna(subset=[y]), x="source", y=y, color="category", title=title,
                         color_discrete_map={"ACTUAL": "#168b85", "ESTIMATED": "#dd9d30", "REFERENCE": "#788aa4"}))
        delta = m["accuracy"] * 100 - float(reference.iloc[0]["accuracy_percent"])
        st.write(f"Our measured result{demo_suffix} differs from the paper's reported AWS value by {delta:+.2f} percentage points.")
        st.warning("The experimental conditions differ, so the values should not be treated as directly equivalent. No architectural winner can be inferred from these figures.")
    elif page == "Experiment Details":
        st.header("Experiment details")
        st.write("Saved automatically to:")
        st.code(result["folder"])
        st.caption("Page navigation reuses the experiment. Disk results are reused when dataset, code, model configuration, hardware identity and library versions match.")
        st.subheader("Automatic resource-aware configuration")
        st.json(cfg)
        st.subheader("Measured timing boundaries")
        cards([("Feature extraction (train + validation)", f"{m['feature_extraction_seconds']:.2f} s"),
               ("Head fitting + validation", f"{m['head_training_seconds']:.2f} s"),
               ("Best validation epoch", str(m["best_epoch"])),
               ("Training image presentations", str(m["training_image_presentations"]))])
        st.json(cloud["pricing"])
        st.download_button("Download complete result JSON", json.dumps(result, indent=2), "experiment_result.json", "application/json")
        for name in ("training_history.csv", "classification_report.csv", "confusion_matrix.csv", "scalability.csv", "split_manifest.csv", "resources.csv"):
            path = Path(result["folder"]) / name
            if path.exists():
                st.download_button(f"Download {name}", path.read_bytes(), name, "text/csv")
else:
    st.info("Experiment unavailable. Dataset information and methodology remain accessible.")
