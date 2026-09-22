"""EfficientNetB0 backbone acquisition, preprocessing, and batched feature extraction."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from PIL import Image
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision import transforms
from backend.utils import ROOT, download


def seed_everything(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, min(4, __import__('os').cpu_count() or 1)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_backbone(device="cpu", root=ROOT):
    seed_everything()
    model = efficientnet_b0(weights=None)
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    target = root / "models" / Path(weights.url).name
    info = {"model": "EfficientNetB0", "pretrained": False, "weights_source": weights.url, "warning": None}
    try:
        if not target.exists():
            download(weights.url, target, max_bytes=40_000_000, deadline=120)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        expected = target.stem.rsplit("-", 1)[-1]
        if not digest.startswith(expected):
            target.unlink(missing_ok=True)
            raise ValueError("Pretrained weight checksum failed")
        model.load_state_dict(torch.load(target, map_location="cpu", weights_only=True))
        info.update(pretrained=True, weights_sha256=digest)
    except Exception as exc:
        info["warning"] = f"Pretrained weights unavailable: {exc}. Random frozen features; NOT transfer learning; demonstration only."
    model.classifier = nn.Identity()
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval().to(device)
    return model, info


PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])


def image_tensor(path):
    with Image.open(path) as image:
        return PREPROCESS(image.convert("RGB"))


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def extract_features(backbone, rows, batch_size, device, progress=lambda message: None):
    features = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = torch.stack([image_tensor(r["path"]) for r in rows[start:start + batch_size]]).to(device)
            features.append(backbone(batch).cpu())
            progress(f"Extracting EfficientNet features: {min(start + batch_size, len(rows))}/{len(rows)} images")
    return torch.cat(features).clone()


def make_head():
    seed_everything()
    return nn.Sequential(nn.Dropout(.2), nn.Linear(1280, 4))
