# Validation record — 22 September 2026

Validated using Python 3.12.4 on Windows 11, the pinned requirements, PyTorch CPU execution, and a 15.35 GiB RAM machine. Dependencies were installed in the project's isolated `.venv`; `pip check` reported no broken requirements.

## Completed checks

- `python -m compileall .`: passed, including the full workspace.
- `pytest`: **38 passed**. Includes real EfficientNet computation on synthetic images, five independently trained scalability runs, all eight Streamlit pages, generated Plotly chart counts, cached initialization across separate sessions, classification metrics, report support counts, efficiency formulas, cost scenarios, live-price parsing/fallback, corruption and duplicate handling, deterministic splits, source failover, and experiment storage.
- Both public repositories were successfully downloaded and decoded without authentication. The final experiment uses the primary `Falah/Alzheimer_MRI` source.
- The primary source contains 6,400 valid images: 896 MildDemented, 64 ModerateDemented, 3,200 NonDemented, and 2,240 VeryMildDemented. No corrupt or exact duplicate images were found in this acquisition.
- Official ImageNet EfficientNetB0 weights downloaded successfully and passed SHA-256 prefix validation.
- Real default application initialization completed through `streamlit.testing.v1.AppTest`; every page executed without errors, and a fresh application session reused its saved experiment.
- `streamlit run app.py --server.port 8501`: server started; `http://127.0.0.1:8501/_stcore/health` returned HTTP 200 and `ok`.
- Live public AWS pricing returned $0.526/hour for the selected reference instance. No cloud infrastructure was provisioned.
- Automated outage tests verify that both public download failures produce a synthetic demo dataset, unavailable weights are explicitly labeled as random-feature demonstration mode, and pricing outages use the bundled table.

## Final measured local run

Source artifact: `experiments/20260922_183213/summary.json` (generated locally and excluded from Git, along with downloaded data and model weights).

The automatic CPU budget selected 1,600 images, batch size 16, and five classifier epochs. Splits contained 1,119 training, 240 validation, and 241 test images. EfficientNetB0 used its pretrained frozen backbone.

| Quantity | Value | Provenance |
|---|---:|---|
| Accuracy | 48.5477% | Actual test predictions |
| Macro precision | 0.367552 | Actual test predictions |
| Macro recall | 0.406478 | Actual test predictions |
| Macro F1 | 0.335799 | Actual test predictions |
| Weighted F1 | 0.487436 | Actual test predictions |
| Training time | 34.1189 seconds | Measured train/validation extraction + head fitting |
| Mean inference latency | 27.6592 ms | Measured after warm-up |
| AWS training scenario cost | $0.00882222 | Estimated compute + temporary EBS; 60-second minimum |
| Project efficiency score | 73.2029/100 | Project formula mixing actual measurements and estimated cost |

All five 10/25/50/75/100% scalability experiments completed, each using one epoch, independently reset heads, re-extracted features, and the same validation/test partition. Their full measurements and artifacts are under that run's `scalability/` directory. The main result was not replaced with a better reference-paper value. Low classification performance demonstrates that this short feature-extraction workload is not a validated clinical classifier.

## Scope limits

No supported browser was connected in this session, so pixel-level browser inspection was unavailable. Page execution and Plotly chart generation were checked with Streamlit's testing framework; HTTP startup was checked separately. CUDA execution was not tested on physical GPU hardware here. Network outage tests used controlled failures; real primary and backup downloads were also exercised.

The paper file was not present, so reference figures remain user-supplied and unverified against the original document. AWS time/latency are uncalibrated scenarios, not measurements. The image-level split cannot rule out patient/slice leakage. This is an educational research prototype, not a diagnostic system. See `METHODOLOGY.md` for full assumptions.
