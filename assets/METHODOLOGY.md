# Methodology and interpretation

This platform measures a **local** four-class image-classification workload. It estimates one AWS scenario; it does not test AWS reliability, network failover, distributed execution or multi-cloud coordination.

## Data and provenance

Primary: [Falah/Alzheimer_MRI](https://huggingface.co/datasets/Falah/Alzheimer_MRI), pinned revision `daac24f9597236b45837d82f7eb9c9ad1f8c60c8`. The publisher lists Apache-2.0. Backup: [Obscure-Entropy/Alzheimer-MRI](https://huggingface.co/datasets/Obscure-Entropy/Alzheimer-MRI), pinned revision `02dc8a193fc452348e45d9192526076f1b11a89c`, publisher-listed MIT. Only the backup's unaugmented train/valid/test files are used. These are community-hosted derivatives, not independent cohorts; publisher metadata does not establish clinical provenance or permissions for every downstream use.

Public Parquet files are downloaded without authentication and embedded images extracted. Numeric-label mappings come from embedded class metadata or, when the export omits metadata, the verified label order in each pinned source's dataset card. Nested class folders are recognized with case/space/underscore normalization. PIL decodes every image; dimensions, class counts, corrupt files, exact pixel duplicates, and conflicting labels are recorded. Exact duplicates are removed before splitting. Failed downloads automatically try the second repository, then bundled synthetic geometric/noise images. Demo data is never presented as MRI or evidence of Alzheimer's classification. Public sources are retried after 24 hours when demo data is cached.

## Split and automatic compute budget

Seed 42 controls class-stratified selection, per-class 70/15/15 train/validation/test allocation (integer rounding recorded), and classifier initialization. CUDA: 100%, 10 epochs, batch 32 (16 below 4 GiB GPU memory). CPU with at least 8 GiB total and 3 GiB available RAM: 50%, 5 epochs, batch 16. Otherwise: 25%, 3 epochs, batch 8. An additional automatic image cap of 12,000/1,600/400 respectively limits initialization cost; rare classes retain at least 10 images. The exact fraction/counts are saved. Runtime is not guaranteed on arbitrary machines. GPU use requires a CUDA-capable PyTorch installation and functioning driver; otherwise CPU runs automatically. Apple MPS is not used.

These are **image-level splits**, not patient-level validation. Patient identifiers are unavailable in these mirrors. Related slices, scans and near-duplicates could cross partitions despite exact deduplication, inflating performance. This is a major limitation. The source's original split is replaced with the requested seeded split. No external clinical validation or patient-level generalization claim is made.

## Model and training measurement

[Torchvision EfficientNetB0](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html) uses ImageNet pretrained weights checked against their published filename hash. Images become RGB, resized to 224×224, and normalized with ImageNet channel statistics. The backbone is frozen, features extracted once per split, and a four-class linear head with dropout 0.2 is trained using Adam (learning rate 0.001) and inverse-frequency class-weighted cross entropy. No synthetic balancing or augmentations are applied. Best validation loss selects the classifier; test data never selects a model. The small head is trained on CPU even when CUDA accelerates the backbone. This is feature-extraction transfer learning, not full-network fine-tuning. If weights cannot be obtained, random frozen features run for demonstration only and all pages show the limitation.

Training time includes train/validation feature extraction, head optimization, and validation selection. It excludes acquisition, pretrained model initialization, held-out evaluation and inference benchmarking. Main epochs reuse extracted features. Saved learning curves are end-of-epoch evaluations with dropout disabled. Epoch durations cover head optimization and validation, while separate extraction and total timings are retained. GPU operations are synchronized. Repeatability is best effort, not bitwise across hardware and library versions.

Accuracy, macro/weighted precision, recall and F1, a four-class confusion matrix, support counts, and classification report are computed on held-out test images. Undefined per-class ratios are zero. Imbalance makes accuracy alone insufficient. Inference includes disk decode (warm OS cache), resize, normalization, device transfer and complete model execution at batch size 1. Three warm-ups precede 32 timed predictions. Mean, median/P50, P95 and reciprocal-mean throughput are reported. This is not network/API latency or batched server throughput.

## Resources and scalability

CPU percent and system RAM pressure, process RSS, and (when available) NVIDIA GPU utilization and GPU memory are sampled approximately every 0.5 seconds during training. GPU counters can be unavailable even with CUDA; missing values stay unavailable. System metrics include other applications. Both means and peaks and raw samples are saved. Brief head epochs may span only a few samples.

Five independent head-training runs use nested 10%, 25%, 50%, 75%, and 100% subsets **of the automatically selected training pool**, retaining at least one image per class. The same validation and test sets are held fixed. Every run starts from the same pretrained backbone and freshly initialized head and **re-extracts** features, so feature extraction time is measured at every size. CPU scalability runs use one epoch; CUDA runs use two, equally for all sizes. These learning-budget experiments are not directly equivalent to the main 3/5/10-epoch result. One run per size has no confidence intervals; repeated test reporting is descriptive and must not be used for tuning. Measurements need not be monotonic. Failed runs say “Experiment unavailable.”

## Estimated AWS performance and cost

Reference: `g4dn.xlarge`, Linux, shared tenancy, On-Demand, `us-east-1`. [AWS public EC2 pricing JSON](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json) is queried anonymously, falling back to a bundled $0.526/hour reference. The pricing source, timestamp and fallback reason are saved. [EBS gp3](https://aws.amazon.com/ebs/pricing/) uses a bundled $0.08/GB-month reference.

Estimated training time = measured local training seconds × 0.25 (CPU) or × 1.0 (CUDA). These are **uncalibrated scenario assumptions**, not validated hardware performance ratios. Estimated latency uses the same multiplier. Sensitivity scenarios halve/double the multiplier; these ranges are not confidence intervals. AWS accuracy is unavailable; accuracy is not copied or improved by this simulation. An arbitrary local GPU may be faster or slower than AWS's T4.

Compute cost = max(60, ceil(estimated seconds)) / 3600 × hourly USD. A gp3 volume of max(30, ceil(dataset GiB)+10) GB is assumed, retained only for that billed window, using a 30-day month. Storage cost = volume GB × $/GB-month × billed seconds / (30×86400). Total = compute + storage. Cost/image divides total by unique training images. Costs exclude setup, download, evaluation, network, idle, tax, snapshots, and long-term storage, and do not sum the scalability suite. This narrow scenario can underestimate a practical deployment. Saved experiments retain their original price snapshot.

**AWS values are estimates based on public pricing and project-defined performance scaling assumptions. No AWS resources were launched. No AWS infrastructure was actually launched.**

## Efficiency definitions

Accuracy uses a fraction from 0 to 1. Training efficiency = accuracy / training seconds. Cost efficiency = accuracy / estimated AWS training cost. Training throughput = unique training images / training seconds; epoch image presentations / training seconds is reported separately and is not CNN throughput because features are reused.

Component scores, 0–100: accuracy = 100×accuracy; time = 100/(1+seconds/600); cost = 100/(1+USD/0.10); latency = 100/(1+milliseconds/50). Fixed project anchors are 600 seconds, $0.10, and 50 ms; these were not fitted to paper values. A score of 50 corresponds to each anchor. Scores depend on the chosen workload/budget and are not fair cross-project rankings.

Compute utility = max(0,100×(1−abs(utilization−65)/65)), using GPU utilization when measurable, otherwise CPU utilization. Memory utility = max(0,100×(1−max(0,RAM%−70)/30)). Resource score = 0.6×compute utility + 0.4×memory utility. This intentionally penalizes saturation and idleness, but is only a heuristic; it does not measure energy efficiency or prove an optimal operating point.

**Project Efficiency Score = 0.30×Accuracy Score + 0.25×Training Time Score + 0.20×Cost Score + 0.15×Latency Score + 0.10×Resource Score.**

This is a project-defined composite efficiency score and is not an industry-standard metric. It mixes local measurements with scenario costs.

## Reference-paper comparison and limitations

The four paper rows in `data/reference_paper_results.csv` reproduce **user-supplied reference values** attributed to “Optimizing Alzheimer's Data Analysis Through Multi-Cloud Integration: A Performance and Reliability Approach.” The paper file was not available in the project workspace; these values have not been independently verified against it. They are never experimental measurements. Dataset, hardware, split policy, training method and timing boundaries may differ. Charts distinguish ACTUAL, ESTIMATED and REFERENCE; they do not establish a winner or that single-cloud is universally preferable.

AWS performance is estimated rather than measured on EC2 hardware. AWS cost uses public pricing and assumptions. Local hardware differs from cloud hardware. Reference paper results use different experimental conditions, so direct numerical comparison has limitations. No cloud reliability experiment is performed. This system is a research/educational prototype, not a clinical diagnostic system.

**Research and educational use only. This system is not a medical diagnostic device.**
