"""Sample system pressure separately from process memory and CUDA allocations."""
from __future__ import annotations
import os
import platform
import shutil
import subprocess
import threading
import time
import numpy as np
import psutil


def detect_hardware():
    import torch
    gpu = torch.cuda.is_available()
    if gpu:
        try:
            torch.zeros(1, device="cuda")
        except RuntimeError:
            gpu = False
    ram = psutil.virtual_memory()
    return {"cpu": platform.processor() or platform.machine(), "cpu_count": psutil.cpu_count(),
            "ram_gb": round(ram.total / 2**30, 2), "available_ram_gb": round(ram.available / 2**30, 2),
            "gpu": torch.cuda.get_device_name(0) if gpu else None,
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if gpu else None,
            "device": "cuda" if gpu else "cpu", "platform": platform.platform()}


def automatic_config(hardware):
    if hardware["device"] == "cuda":
        fraction, epochs, batch, cap = 1.0, 10, 32, 12000
        if hardware["gpu_memory_gb"] < 4:
            batch = 16
    elif hardware["ram_gb"] >= 8 and hardware["available_ram_gb"] >= 3:
        fraction, epochs, batch, cap = .50, 5, 16, 1600
    else:
        fraction, epochs, batch, cap = .25, 3, 8, 400
    return {"seed": 42, "model": "EfficientNetB0", "image_size": 224, "epochs": epochs,
            "batch_size": batch, "dataset_fraction": fraction, "max_images": cap,
            "learning_rate": .001, "device": hardware["device"], "scalability_epochs": 2 if hardware["device"] == "cuda" else 1,
            "strategy": "Frozen ImageNet backbone; cached features; train a new linear classifier",
            "budget_note": "CPU dataset cap limits initialization time; minimum 10 images/class overrides the cap if necessary."}


class ResourceMonitor:
    def __init__(self, device="cpu", interval=.5):
        self.device, self.interval = device, interval
        self.samples = []
        self.stop_event = threading.Event()
        self.process = psutil.Process()
        self.smi = shutil.which("nvidia-smi") if device == "cuda" else None

    def sample(self):
        memory = psutil.virtual_memory()
        row = {"timestamp": time.time(), "cpu_percent": psutil.cpu_percent(),
               "ram_percent": memory.percent, "ram_used_mb": memory.used / 2**20,
               "process_ram_mb": self.process.memory_info().rss / 2**20,
               "gpu_percent": None, "gpu_memory_mb": None}
        if self.smi:
            try:
                args = [self.smi, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "--id=0"]
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                text = subprocess.check_output(args, timeout=2, creationflags=flags).decode().strip()
                row["gpu_percent"], row["gpu_memory_mb"] = [float(x.strip()) for x in text.split(",")]
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        self.samples.append(row)

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def __enter__(self):
        psutil.cpu_percent()
        self.sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=4)
        self.sample()

    def summary(self):
        result = {"sample_count": len(self.samples), "scope": "System CPU/RAM/GPU, process RSS; training phase only"}
        for key in ("cpu_percent", "ram_percent", "ram_used_mb", "process_ram_mb", "gpu_percent", "gpu_memory_mb"):
            values = [r[key] for r in self.samples if r[key] is not None]
            result[key] = float(np.mean(values)) if values else None
            result[key + "_peak"] = max(values) if values else None
        return result
