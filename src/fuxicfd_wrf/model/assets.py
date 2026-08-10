"""Locate and fetch the upstream FuXi-CFD assets (ONNX weights, scalers, examples).

Assets live outside the repo -- they are 889 MB and CC BY-NC 4.0 licensed, so we
never vendor them. Default location is $FUXICFD_ASSETS, falling back to
/scratch/$USER/fuxicfd/assets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

HF_BASE = (
    "https://huggingface.co/linchensen/FuXi-CFD-model/resolve/main/fuxicfd-model"
)

#: Local relative path -> upstream path within the HF repo.
ASSET_FILES = {
    "model/fuxicfd_model.onnx": "model/fuxicfd_model.onnx",
    "normalization/scaler_input.npy": "inference_example/normalization/scaler_input.npy",
    "normalization/scaler_output.npy": "inference_example/normalization/scaler_output.npy",
    "utils/height.npy": "inference_example/utils/height.npy",
    "example/inputs.npz": "inference_example/data/inputs.npz",
    "example/prediction.npz": "inference_example/data/prediction.npz",
    "example/outputs.npz": "inference_example/data/outputs.npz",
}


def default_assets_root() -> Path:
    env = os.environ.get("FUXICFD_ASSETS")
    if env:
        return Path(env)
    return Path(f"/scratch/{os.environ.get('USER', 'nobody')}/fuxicfd/assets")


def default_cache_root() -> Path:
    env = os.environ.get("FUXICFD_CACHE")
    if env:
        return Path(env)
    return Path(f"/scratch/{os.environ.get('USER', 'nobody')}/fuxicfd/cache")


@dataclass(frozen=True)
class Assets:
    """Resolved paths to the upstream FuXi-CFD assets."""

    root: Path

    @property
    def onnx(self) -> Path:
        return self.root / "model/fuxicfd_model.onnx"

    @property
    def scaler_input(self) -> Path:
        return self.root / "normalization/scaler_input.npy"

    @property
    def scaler_output(self) -> Path:
        return self.root / "normalization/scaler_output.npy"

    @property
    def height(self) -> Path:
        return self.root / "utils/height.npy"

    @property
    def example_inputs(self) -> Path:
        return self.root / "example/inputs.npz"

    @property
    def example_prediction(self) -> Path:
        return self.root / "example/prediction.npz"

    @property
    def example_outputs(self) -> Path:
        return self.root / "example/outputs.npz"

    def missing(self) -> list[str]:
        return [rel for rel in ASSET_FILES if not (self.root / rel).exists()]

    def require(self, *paths: Path) -> None:
        absent = [p for p in paths if not p.exists()]
        if absent:
            raise FileNotFoundError(
                "Missing FuXi-CFD assets:\n"
                + "\n".join(f"  {p}" for p in absent)
                + "\n\nRun: fuxicfd-wrf fetch-model"
            )

    def load_input_scaler(self) -> dict:
        self.require(self.scaler_input)
        return np.load(self.scaler_input, allow_pickle=True).item()

    def load_output_scaler(self) -> dict:
        self.require(self.scaler_output)
        return np.load(self.scaler_output, allow_pickle=True).item()


def resolve(root: Path | str | None = None) -> Assets:
    return Assets(Path(root) if root is not None else default_assets_root())


def fetch(root: Path | str | None = None, *, force: bool = False) -> Assets:
    """Download any missing upstream assets into `root`.

    Downloads stream to a .part file and are renamed on completion, so an
    interrupted fetch never leaves a truncated file that looks valid.
    """
    assets = resolve(root)
    for rel, remote in ASSET_FILES.items():
        dest = assets.root / rel
        if dest.exists() and not force:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        url = f"{HF_BASE}/{remote}"
        print(f"[fetch] {rel} <- {url}")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 22):
                    fh.write(chunk)
        tmp.rename(dest)
    return assets
