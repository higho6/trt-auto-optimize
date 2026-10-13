import argparse
from pathlib import Path

import onnx
import torch
from onnxsim import simplify
from transformers import SiglipVisionModel


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NAMES = ["last_hidden_state", "pooler_output"]


class VisionExport(torch.nn.Module):
    def __init__(self, model: SiglipVisionModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        return outputs.last_hidden_state, outputs.pooler_output


def verify_export(
    model: VisionExport, output: Path, image_size: int,
    batch_size: int | None = None,
) -> None:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(output), providers=["CPUExecutionProvider"]
    )
    batch_sizes = (1, 2) if batch_size is None else (batch_size,)
    for batch_size in batch_sizes:
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        with torch.inference_mode():
            expected = model(pixel_values)
        actual = session.run(
            OUTPUT_NAMES, {"pixel_values": pixel_values.numpy()}
        )
        for name, reference, result in zip(OUTPUT_NAMES, expected, actual):
            np.testing.assert_allclose(
                result, reference.numpy(), rtol=1e-3, atol=1e-4,
                err_msg=f"{name}: batch_size={batch_size}",
            )
        print(f"ONNX Runtime verification passed: batch_size={batch_size}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the SigLIP vision model to ONNX (FP32)."
    )
    parser.add_argument(
        "--model-path", type=Path,
        default=ROOT / "ckpt" / "siglip-base-patch16-224",
        help="Local Hugging Face checkpoint directory.",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "onnx" / "siglip_vision.onnx",
        help="Destination ONNX file.",
    )
    parser.add_argument(
        "--batch-size", type=int,
        help="Export a static positive batch size; omit for dynamic batch.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Compare ONNX Runtime and PyTorch outputs (dynamic: 1 and 2; static: selected batch).",
    )
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")

    vision_model = SiglipVisionModel.from_pretrained(
        str(args.model_path), local_files_only=True,
        attn_implementation="eager",
    ).cpu().float().eval()
    model = VisionExport(vision_model).eval()
    image_size = vision_model.config.image_size
    export_batch_size = 1 if args.batch_size is None else args.batch_size
    pixel_values = torch.randn(export_batch_size, 3, image_size, image_size)
    dynamic_axes = None
    if args.batch_size is None:
        dynamic_axes = {
            "pixel_values": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"},
            "pooler_output": {0: "batch_size"},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (pixel_values,),
            str(args.output),
            input_names=["pixel_values"],
            output_names=OUTPUT_NAMES,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
    onnx.checker.check_model(str(args.output))
    onnx_model = onnx.load(str(args.output))
    original_nodes = len(onnx_model.graph.node)
    check_batch_size = 2 if args.batch_size is None else args.batch_size
    onnx_model, valid = simplify(
        onnx_model,
        check_n=1,
        skipped_optimizers=["fuse_attention"],
        test_input_shapes={
            "pixel_values": [check_batch_size, 3, image_size, image_size]
        },
        check_rtol=1e-3,
        check_atol=1e-4,
    )
    if not valid:
        raise RuntimeError("ONNX simplification equivalence check failed.")
    onnx_model = onnx.shape_inference.infer_shapes(
        onnx_model, check_type=True, strict_mode=True, data_prop=True,
    )
    onnx.checker.check_model(onnx_model, full_check=True)
    onnx.save(onnx_model, str(args.output))
    print(
        f"ONNX simplified: {original_nodes} -> {len(onnx_model.graph.node)} nodes; "
        f"shape inference completed: {len(onnx_model.graph.value_info)} intermediate tensors"
    )
    if args.verify:
        verify_export(model, args.output, image_size, args.batch_size)
    print(f"ONNX exported and checked: {args.output}")


if __name__ == "__main__":
    main()