"""Create equivalent reference ONNX and a KdaScale custom-node integration test."""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shape = ["batch", 16]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)]
    reference = helper.make_model(helper.make_graph([
        helper.make_node("Mul", ["x", "scale"], ["y"])
    ], "scale_reference", inputs, outputs, [
        helper.make_tensor("scale", TensorProto.FLOAT, [], [1.25])
    ]), opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    plugin = helper.make_model(helper.make_graph([
        helper.make_node(
            "KdaScale", ["x"], ["y"], name="optimized_scale", domain="kda",
            scale=1.25, plugin_version="1", plugin_namespace="",
        )
    ], "scale_plugin", inputs, outputs), opset_imports=[
        helper.make_opsetid("", 17), helper.make_opsetid("kda", 1)
    ], ir_version=9)
    for name, model in (("reference.onnx", reference), ("plugin.onnx", plugin)):
        onnx.checker.check_model(model)
        onnx.save(model, str(args.output_dir / name))
    print(f"Reference/plugin ONNX saved to {args.output_dir}; scale example is not an optimization.")


if __name__ == "__main__":
    main()
