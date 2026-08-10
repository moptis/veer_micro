"""onnxruntime session wrapper with batch-size handling.

The shipped ``model_config.yml`` advertises a fixed input shape of
[1, 4, 300, 300]. Whether the exported graph actually pins the batch axis or
leaves it symbolic determines whether we can batch on the GPU at all, so we
introspect it at load time rather than assuming either way.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..constants import MODEL_SHAPE, N_LEVELS, OUTPUT_VARS

#: Hard cap on batch size.
#:
#: At batch 32 the final `/patch_unembed/upconv/Conv` fails on H100 with
#: CUDNN_STATUS_NOT_SUPPORTED -- the output tensor (32 x 27 x 4 x 300 x 300 =
#: 311M elements) overflows what cuDNN will describe in one descriptor. Measured
#: throughput saturates at batch 4 anyway (~42 tile-steps/s at batch 4, 8 and 16),
#: so there is nothing to gain by going higher.
MAX_SAFE_BATCH = 16


def _allocated_cpus() -> int:
    """CPUs this process may actually use.

    Prefers Slurm's allocation, then the process affinity mask, then the machine
    core count. `os.cpu_count()` reports all 128 cores of a Kestrel node even when
    Slurm gave the task 4, which is how you end up oversubscribed.
    """
    slurm = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm and slurm.isdigit() and int(slurm) > 0:
        return int(slurm)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def available_providers() -> list[str]:
    import onnxruntime as ort

    return list(ort.get_available_providers())


def _pick_providers(device: str) -> list[str]:
    avail = available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" not in avail:
        if device == "cuda":
            raise RuntimeError(
                "device='cuda' requested but CUDAExecutionProvider is unavailable.\n"
                f"onnxruntime reports: {avail}\n"
                "Install onnxruntime-gpu (pip install 'fuxicfd-wrf[gpu]') and make "
                "sure you are on a GPU node."
            )
        return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


class FuxiCFDSession:
    """Wraps an ONNX session and hides whether the graph supports batching.

    Call `run(x)` with (B, 4, 300, 300); you always get (B, 27, 4, 300, 300) back,
    regardless of whether the underlying graph batches natively or has to loop.
    """

    def __init__(
        self,
        onnx_path: Path | str,
        *,
        device: str = "auto",
        intra_op_threads: int | None = None,
    ):
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\nRun: fuxicfd-wrf fetch-model"
            )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Default to the cpuset we were actually given, not the machine's core
        # count. Under Slurm those differ, and onnxruntime otherwise spawns a
        # thread per physical core and floods stderr with pthread_setaffinity_np
        # failures as it tries to pin threads to CPUs outside the allocation.
        opts.intra_op_num_threads = int(intra_op_threads or _allocated_cpus())

        self.providers = _pick_providers(device)
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=self.providers
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = list(inp.shape)
        self.output_name = self.session.get_outputs()[0].name

        # A symbolic (str) or negative first dim means the batch axis is dynamic.
        first = self.input_shape[0] if self.input_shape else None
        self.dynamic_batch = isinstance(first, str) or (
            isinstance(first, int) and first < 1
        )
        self.fixed_batch = None if self.dynamic_batch else int(first)

    @property
    def active_provider(self) -> str:
        return self.session.get_providers()[0]

    @property
    def on_gpu(self) -> bool:
        return self.active_provider == "CUDAExecutionProvider"

    def max_batch(self, requested: int) -> int:
        """Largest usable batch: `requested`, clamped to what the graph and cuDNN allow."""
        if not self.dynamic_batch:
            return self.fixed_batch or 1
        batch = max(1, requested)
        if batch > MAX_SAFE_BATCH:
            print(
                f"[session] batch_size={batch} clamped to {MAX_SAFE_BATCH}: larger "
                "batches fail in cuDNN at the final upconv, and throughput already "
                "saturates by batch 4."
            )
            batch = MAX_SAFE_BATCH
        return batch

    def run(self, x: np.ndarray) -> np.ndarray:
        """Run inference on (B, 4, 300, 300) -> (B, 27, 4, 300, 300)."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 4 or x.shape[1:] != (4, *MODEL_SHAPE):
            raise ValueError(f"expected (B,4,{MODEL_SHAPE[0]},{MODEL_SHAPE[1]}), got {x.shape}")

        if self.dynamic_batch or x.shape[0] == self.fixed_batch:
            out = self.session.run([self.output_name], {self.input_name: x})[0]
        else:
            # Graph pins the batch axis: fall back to a loop. Slower, but correct.
            chunks = [
                self.session.run([self.output_name], {self.input_name: x[i : i + 1]})[0]
                for i in range(x.shape[0])
            ]
            out = np.concatenate(chunks, axis=0)

        expected = (x.shape[0], N_LEVELS, len(OUTPUT_VARS), *MODEL_SHAPE)
        if out.shape != expected:
            raise RuntimeError(f"model returned {out.shape}, expected {expected}")
        return out

    def describe(self) -> str:
        batch = "dynamic" if self.dynamic_batch else f"fixed={self.fixed_batch}"
        return (
            f"FuXiCFDSession(provider={self.active_provider}, batch={batch}, "
            f"input={self.input_name}{self.input_shape})"
        )
