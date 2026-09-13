from __future__ import annotations

import math
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Protocol, Sequence

from PIL import Image


class ClipRuntime(Protocol):
    def encode_image(self, image_path: Path) -> Sequence[float]: ...


RuntimeLoader = Callable[[str, str], ClipRuntime]


def resolve_device(requested: str, torch_module: Any | None = None) -> str:
    if requested != "auto":
        return requested
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return "cpu"
    return "cuda" if torch_module.cuda.is_available() else "cpu"


class LocalClipEncoder:
    def __init__(
        self,
        *,
        model_name: str = "openai/clip-vit-base-patch32",
        dimension: int = 512,
        device: str = "auto",
        loader: RuntimeLoader | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_version = model_name
        self.dimension = dimension
        self.requested_device = device
        self._loader = loader or _load_transformers_runtime
        self._runtime: ClipRuntime | None = None
        self._load_lock = Lock()

    def encode_image(self, image_path: str | Path) -> list[float]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        vector = [float(value) for value in self._ensure_runtime().encode_image(path)]
        if len(vector) != self.dimension:
            raise ValueError(f"expected {self.dimension} dimensions, got {len(vector)}")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise ValueError("CLIP returned a zero vector")
        return [value / norm for value in vector]

    def _ensure_runtime(self) -> ClipRuntime:
        if self._runtime is not None:
            return self._runtime
        with self._load_lock:
            if self._runtime is None:
                device = resolve_device(self.requested_device)
                self._runtime = self._loader(self.model_name, device)
        return self._runtime


class _TransformersClipRuntime:
    def __init__(self, model: Any, processor: Any, *, device: str, torch_module: Any) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.torch = torch_module

    def encode_image(self, image_path: Path) -> list[float]:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with self.torch.inference_mode():
            features = self.model.get_image_features(**inputs)
        return features[0].detach().float().cpu().tolist()


def _load_transformers_runtime(model_name: str, device: str) -> ClipRuntime:
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Local CLIP dependencies are missing; install the 'multimodal' optional dependencies"
        ) from exc
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return _TransformersClipRuntime(model, processor, device=device, torch_module=torch)
