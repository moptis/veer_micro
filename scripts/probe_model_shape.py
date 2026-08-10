"""Probe whether FuXi-CFD's 300x300 spatial size is a hard constraint.

Answers, empirically rather than by reading the paper:
  1. Are the spatial dims symbolic or literal in the exported graph?
  2. Does the architecture bake the grid size into weights (positional
     embeddings, hardcoded Reshape targets)?
  3. Does the model actually run if the declared dims are relaxed?

    python scripts/probe_model_shape.py [--try-size 150]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx

from fuxicfd_wrf.model import assets as assets_mod

#: Sizes that would appear in a weight shape if the grid size were baked in.
GRID_DERIVED = {300, 150, 75, 38, 90000, 22500, 5625, 1444}


def dims(vi):
    out = []
    for d in vi.type.tensor_type.shape.dim:
        out.append(d.dim_param if d.HasField("dim_param") else d.dim_value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--try-size", type=int, default=150)
    args = ap.parse_args()

    path = assets_mod.resolve().onnx
    model = onnx.load(str(path), load_external_data=False)
    g = model.graph

    print("=== declared signature ===")
    for v in g.input:
        print(f"  input  {v.name}: {dims(v)}")
    for v in g.output:
        print(f"  output {v.name}: {dims(v)}")

    print("\n=== weights whose shape encodes the grid size ===")
    baked = [
        (i.name, list(i.dims)) for i in g.initializer
        if any(d in GRID_DERIVED for d in i.dims)
    ]
    for name, shape in baked[:15]:
        print(f"  {name:65s} {shape}")
    if len(baked) > 15:
        print(f"  ... and {len(baked) - 15} more")
    print(f"  TOTAL: {len(baked)} grid-dependent weight tensors")

    print("\n=== hardcoded Reshape targets containing grid sizes ===")
    consts = {i.name: onnx.numpy_helper.to_array(i) for i in g.initializer}
    hard = 0
    for node in g.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        tgt = consts.get(node.input[1])
        if tgt is not None and any(int(v) in GRID_DERIVED for v in np.ravel(tgt)):
            hard += 1
            if hard <= 8:
                print(f"  {node.name or node.op_type:55s} -> {np.ravel(tgt).tolist()}")
    print(f"  TOTAL: {hard} Reshape nodes with a baked-in spatial size")

    print("\n=== op mix (top 12) ===")
    for op, n in Counter(nd.op_type for nd in g.node).most_common(12):
        print(f"  {op:20s} {n}")

    print(f"\n=== empirical: relax declared dims to {args.try_size} and run ===")
    import onnxruntime as ort

    relaxed = onnx.load(str(path))
    for v in list(relaxed.graph.input) + list(relaxed.graph.output):
        shp = v.type.tensor_type.shape
        for d in list(shp.dim)[-2:]:
            d.ClearField("dim_value")
            d.dim_param = "spatial"
    tmp = Path("/scratch/moptis/fuxicfd/assets/model/_relaxed.onnx")
    onnx.save(relaxed, str(tmp), save_as_external_data=False)
    try:
        sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
        x = np.zeros((1, 4, args.try_size, args.try_size), dtype=np.float32)
        out = sess.run(None, {"input": x})[0]
        print(f"  RAN. output shape {out.shape}")
    except Exception as exc:
        msg = str(exc).split("\n")[0][:400]
        print(f"  FAILED: {msg}")
    finally:
        tmp.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
