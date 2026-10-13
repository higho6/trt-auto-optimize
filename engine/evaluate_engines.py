"""Independent correctness and paired GPU-latency gate for TensorRT candidates."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch

from check_plugin import load_plugin_library


LOGGER = trt.Logger(trt.Logger.WARNING)


class EngineRunner:
    def __init__(
        self, path: Path, shapes: dict[str, list[int]], plugins: list[Path]
    ) -> None:
        self.logger = LOGGER
        trt.init_libnvinfer_plugins(self.logger, "")
        self.libraries = [load_plugin_library(path) for path in plugins]
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize engine: {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Cannot create execution context: {path}")
        self.stream = torch.cuda.Stream()
        self.inputs: dict[str, torch.Tensor] = {}
        self.outputs: dict[str, torch.Tensor] = {}
        names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
        ]
        input_names = {
            name for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        }
        if input_names != shapes.keys():
            raise ValueError(f"Input names do not match: {input_names} vs {shapes.keys()}")
        for name in names:
            if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError(f"Host I/O tensors are not supported: {name}")
            if self.engine.is_shape_inference_io(name):
                raise ValueError(f"Shape-tensor I/O is not supported: {name}")
            if self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise ValueError(f"Non-linear I/O tensor format is not supported: {name}")
        for name, shape in shapes.items():
            if self.engine.get_tensor_dtype(name) != trt.float32:
                raise ValueError(f"Only FP32 external inputs are supported: {name}")
            if not self.context.set_input_shape(name, tuple(shape)):
                raise ValueError(f"Engine rejected input shape {name}: {shape}")
        missing = self.context.infer_shapes()
        if missing:
            raise ValueError(f"Insufficiently specified tensors: {missing}")
        dtypes = {trt.float32: torch.float32, trt.float16: torch.float16}
        for name in names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dim <= 0 for dim in shape):
                raise ValueError(f"Unresolved or empty tensor shape: {name}: {shape}")
            dtype = self.engine.get_tensor_dtype(name)
            if dtype not in dtypes:
                raise ValueError(f"Unsupported external tensor dtype: {name}: {dtype}")
            tensor = torch.empty(shape, dtype=dtypes[dtype], device="cuda")
            target = self.inputs if name in input_names else self.outputs
            target[name] = tensor
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"Cannot bind tensor address: {name}")

    def enqueue(self) -> None:
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        with torch.cuda.stream(self.stream):
            for name, value in inputs.items():
                self.inputs[name].copy_(torch.from_numpy(value))
            self.enqueue()
        self.stream.synchronize()
        return {
            name: tensor.cpu().float().numpy()
            for name, tensor in self.outputs.items()
        }

    def latency(self, iterations: int) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(self.stream)
        for _ in range(iterations):
            self.enqueue()
        end.record(self.stream)
        end.synchronize()
        return start.elapsed_time(end) / iterations


def promotion_decision(
    baseline: list[float], candidate: list[float], min_speedup: float
) -> dict[str, object]:
    if (
        not baseline or len(baseline) != len(candidate)
        or any(not math.isfinite(x) or x <= 0 for x in baseline + candidate)
        or not math.isfinite(min_speedup) or min_speedup <= 1
    ):
        raise ValueError("Invalid paired benchmark samples or speedup threshold")
    speedup = statistics.median(baseline) / statistics.median(candidate)
    wins = sum(after < before for before, after in zip(baseline, candidate))
    accepted = speedup >= min_speedup and wins >= math.ceil(len(baseline) * 0.8)
    return {
        "promoted": accepted,
        "speedup": speedup,
        "paired_wins": wins,
        "min_speedup": min_speedup,
        "baseline_ms": baseline,
        "candidate_ms": candidate,
        "reason": (
            "Correctness passed; speedup threshold and >=80% paired wins passed."
            if accepted else "Correctness passed, but the performance gate failed."
        ),
    }


def evaluate(
    onnx_path: Path,
    baseline_path: Path,
    candidate_path: Path,
    shapes: dict[str, list[int]],
    plugins: list[Path],
    rtol: float,
    atol: float,
    min_speedup: float,
    iterations: int,
    repeats: int,
) -> dict[str, object]:
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    names = [item.name for item in session.get_outputs()]
    baseline = EngineRunner(baseline_path, shapes, plugins)
    candidate = EngineRunner(candidate_path, shapes, plugins)
    errors = []
    for seed in range(3):
        generator = np.random.default_rng(seed)
        inputs = {
            name: generator.standard_normal(shape, dtype=np.float32)
            for name, shape in shapes.items()
        }
        expected = dict(zip(names, session.run(names, inputs)))
        for label, runner in (("baseline", baseline), ("candidate", candidate)):
            actual = runner.run(inputs)
            if actual.keys() != expected.keys():
                raise ValueError(f"{label} output names differ from original ONNX")
            for name, reference in expected.items():
                result = actual[name]
                if result.shape != reference.shape:
                    raise ValueError(f"{label} output shape mismatch: {name}")
                if not np.isfinite(result).all() or not np.isfinite(reference).all():
                    raise ValueError(f"{label} non-finite output: {name}")
                np.testing.assert_allclose(
                    result, reference, rtol=rtol, atol=atol,
                    err_msg=f"{label}: {name}, seed={seed}",
                )
                errors.append({
                    "engine": label, "output": name, "seed": seed,
                    "max_abs_error": float(np.max(np.abs(result - reference))),
                })
    for runner in (baseline, candidate):
        for _ in range(20):
            runner.enqueue()
        runner.stream.synchronize()
    baseline_ms = []
    candidate_ms = []
    for repeat in range(repeats):
        order = (baseline, candidate) if repeat % 2 == 0 else (candidate, baseline)
        measured = {runner: runner.latency(iterations) for runner in order}
        baseline_ms.append(measured[baseline])
        candidate_ms.append(measured[candidate])
    result = promotion_decision(baseline_ms, candidate_ms, min_speedup)
    baseline_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    candidate_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    if baseline_hash == candidate_hash:
        result.update({
            "promoted": False,
            "reason": "Identical engine binaries; timing noise is not an optimization.",
        })
    result.update({
        "correctness_passed": True,
        "errors": errors,
        "shapes": shapes,
        "rtol": rtol,
        "atol": atol,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "tensorrt": trt.__version__,
        "baseline_sha256": baseline_hash,
        "candidate_sha256": candidate_hash,
        "timing_scope": "Whole-engine CUDA events, no H2D/D2H, no CUDA graph or profiler.",
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--shapes", required=True, help='JSON, e.g. {"pixel_values":[3,3,224,224]}')
    parser.add_argument("--plugin", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--min-speedup", type=float, default=1.03)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.iterations <= 0 or args.repeats < 3:
        parser.error("iterations must be positive and repeats must be >=3")
    if (
        not all(math.isfinite(x) for x in (args.rtol, args.atol, args.min_speedup))
        or args.rtol < 0 or args.atol < 0 or args.min_speedup <= 1
    ):
        parser.error("tolerances must be finite/non-negative and min-speedup must be >1")
    shapes = json.loads(args.shapes)
    if (
        not isinstance(shapes, dict) or not shapes
        or any(
            not isinstance(name, str) or not isinstance(shape, list) or not shape
            or any(type(dim) is not int or dim <= 0 for dim in shape)
            for name, shape in shapes.items()
        )
    ):
        parser.error("shapes must map input names to non-empty positive integer dimensions")
    try:
        result = evaluate(
            args.onnx, args.baseline, args.candidate, shapes, args.plugin,
            args.rtol, args.atol, args.min_speedup, args.iterations, args.repeats,
        )
    except (AssertionError, ValueError, RuntimeError) as error:
        result = {
            "promoted": False, "correctness_passed": False,
            "reason": str(error),
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        raise
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
