from __future__ import annotations

import math
from pathlib import Path

import pytest
from PIL import Image

from gustobot.application.multimodal.encoders import LocalClipEncoder, resolve_device


class FakeRuntime:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[Path] = []

    def encode_image(self, image_path: Path) -> list[float]:
        self.calls.append(image_path)
        return list(self.vector)


def test_local_clip_encoder_loads_runtime_once_and_normalizes(tmp_path: Path) -> None:
    image_path = tmp_path / "dish.jpg"
    Image.new("RGB", (32, 32), color=(20, 40, 60)).save(image_path)
    runtime = FakeRuntime([3.0, 4.0])
    loads: list[tuple[str, str]] = []

    def loader(model_name: str, device: str) -> FakeRuntime:
        loads.append((model_name, device))
        return runtime

    encoder = LocalClipEncoder(
        model_name="test/clip",
        dimension=2,
        device="cpu",
        loader=loader,
    )

    first = encoder.encode_image(image_path)
    second = encoder.encode_image(image_path)

    assert first == pytest.approx([0.6, 0.8])
    assert second == pytest.approx(first)
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0)
    assert loads == [("test/clip", "cpu")]
    assert runtime.calls == [image_path, image_path]
    assert encoder.model_version == "test/clip"


def test_local_clip_encoder_rejects_missing_image(tmp_path: Path) -> None:
    encoder = LocalClipEncoder(loader=lambda model_name, device: FakeRuntime([1.0]))

    with pytest.raises(FileNotFoundError):
        encoder.encode_image(tmp_path / "missing.jpg")


def test_local_clip_encoder_rejects_wrong_vector_dimension(tmp_path: Path) -> None:
    image_path = tmp_path / "dish.png"
    Image.new("RGB", (16, 16)).save(image_path)
    encoder = LocalClipEncoder(dimension=3, loader=lambda model_name, device: FakeRuntime([1.0, 2.0]))

    with pytest.raises(ValueError, match="expected 3 dimensions"):
        encoder.encode_image(image_path)


def test_local_clip_encoder_rejects_zero_vector(tmp_path: Path) -> None:
    image_path = tmp_path / "dish.png"
    Image.new("RGB", (16, 16)).save(image_path)
    encoder = LocalClipEncoder(dimension=2, loader=lambda model_name, device: FakeRuntime([0.0, 0.0]))

    with pytest.raises(ValueError, match="zero vector"):
        encoder.encode_image(image_path)


def test_resolve_device_uses_cuda_only_when_available() -> None:
    class AvailableCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class UnavailableCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class TorchWithCuda:
        cuda = AvailableCuda()

    class TorchWithoutCuda:
        cuda = UnavailableCuda()

    assert resolve_device("auto", TorchWithCuda()) == "cuda"
    assert resolve_device("auto", TorchWithoutCuda()) == "cpu"
    assert resolve_device("cpu", TorchWithCuda()) == "cpu"
