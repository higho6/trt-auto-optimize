"""Find topology-based Transformer fusion review candidates without rewriting ONNX."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import onnx


def analyze(
    model: onnx.ModelProto, layers: list[dict], profile: list[dict]
) -> dict:
    nodes = list(model.graph.node)
    consumers: dict[str, list[int]] = defaultdict(list)
    graph_outputs = {value.name for value in model.graph.output}
    for index, node in enumerate(nodes):
        for value in set(node.input):
            consumers[value].append(index)
    shapes = {
        value.name: [
            dim.dim_param or (dim.dim_value if dim.HasField("dim_value") else None)
            for dim in value.type.tensor_type.shape.dim
        ]
        for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]
        if value.type.HasField("tensor_type") and value.type.tensor_type.HasField("shape")
    }
    weights = {value.name: list(value.dims) for value in model.graph.initializer}
    constant_names = set(weights) | {
        value for node in nodes if node.op_type == "Constant" and node.domain in ("", "ai.onnx")
        for value in node.output
    }
    shapes.update(weights)
    times = {row["name"]: row["averageMs"] for row in profile if "name" in row}
    candidates = []
    seen: set[tuple[str, tuple[int, ...]]] = set()

    def attributes(node) -> dict:
        return {
            attr.name: onnx.helper.get_attribute_value(attr)
            for attr in node.attribute if attr.type in (
                onnx.AttributeProto.INT, onnx.AttributeProto.FLOAT,
                onnx.AttributeProto.INTS,
            )
        }

    def next_node(index: int, types: set[str]) -> int | None:
        node = nodes[index]
        if not node.output:
            return None
        following = consumers[node.output[0]]
        if len(following) != 1 or node.output[0] in graph_outputs:
            return None
        target = following[0]
        return target if nodes[target].domain in ("", "ai.onnx") and nodes[target].op_type in types else None

    def add(kind: str, indices: list[int], notes: list[str], parallel: bool = False) -> None:
        key = (kind, tuple(sorted(indices)))
        if key in seen:
            return
        seen.add(key)
        group = set(indices)
        internal = {
            value for index in indices for value in nodes[index].output
            if any(target in group for target in consumers[value])
        }
        blockers = [
            f"Live intermediate must be retained: {value}"
            for value in internal if value in graph_outputs
            or any(target not in group for target in consumers[value])
        ]
        inputs = sorted({
            value for index in indices for value in nodes[index].input
            if value and not any(value in nodes[source].output for source in indices)
        })
        outputs = sorted({
            value for index in indices for value in nodes[index].output
            if value in graph_outputs or not consumers[value]
            or any(target not in group for target in consumers[value])
        })
        matched = []
        for layer in layers:
            text = layer["Name"] + " " + json.dumps(layer.get("Metadata", ""))
            if any(
                nodes[index].name and re.search(
                    r"(?<![A-Za-z0-9])" + re.escape(nodes[index].name) + r"(?![A-Za-z0-9])", text
                )
                for index in indices
            ):
                matched.append(layer["Name"])
        candidates.append({
            "kind": kind, "status": "review_only",
            "nodes": [
                {"index": index, "name": nodes[index].name, "op": nodes[index].op_type,
                 "attributes": attributes(nodes[index])}
                for index in indices
            ],
            "inputs": [{"name": name, "shape": shapes.get(name)} for name in inputs],
            "outputs": [{"name": name, "shape": shapes.get(name)} for name in outputs],
            "blockers": blockers,
            "semantic_checks": notes,
            "parallel_projections": parallel,
            "engine_layer_hints": matched,
            "mapped_profile_ms": sum(times[name] for name in set(matched) if name in times) if matched else None,
            "mapping_warning": "Name/metadata hints only; overlapping execution layers are not additive speedup.",
        })

    projections: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        if node.domain not in ("", "ai.onnx"):
            continue
        if node.op_type in ("Gemm", "MatMul"):
            if len(node.input) >= 2 and len(weights.get(node.input[1], [])) == 2:
                projections[node.input[0]].append(index)
            split = next_node(index, {"Split"})
            if (
                split is not None and len(nodes[split].output) == 3
                and len(weights.get(node.input[1], [])) == 2
            ):
                group = [index, split]
                for value in nodes[split].output:
                    if len(consumers[value]) == 1:
                        target = consumers[value][0]
                        if nodes[target].op_type == "Add" and nodes[target].domain in ("", "ai.onnx"):
                            group.append(target)
                add("packed_qkv_epilogue_review", group, [
                    "Projection is already packed in ONNX; do not propose packing QKV again.",
                    "Verify the three splits really are Q/K/V and preserve split sizes/order and branch biases.",
                    "Review packed-bias epilogue and consumer layout without breaking existing TensorRT fusion.",
                ])
            prefix = [index]
            following = next_node(index, {"Add", "Reshape", "Identity"})
            while following is not None and nodes[following].op_type in ("Reshape", "Identity"):
                prefix.append(following)
                following = next_node(following, {"Add", "Reshape", "Identity"})
            if following is not None:
                source_output = nodes[prefix[-1]].output[0]
                other = [value for value in nodes[following].input if value != source_output]
                kind = "gemm_bias_epilogue" if other and other[0] in constant_names else "gemm_residual_epilogue"
                group = prefix + [following]
                activation = next_node(following, {"Gelu", "Relu", "Sigmoid", "Tanh"})
                if activation is not None:
                    group.append(activation)
                add(kind, group, [
                    "Verify broadcasting, Gemm alpha/beta/transposes, intervening reshape/layout and exact activation semantics.",
                    "Check whether TensorRT already implements this epilogue.",
                ])
            frontier = [index]
            activation_group = {index}
            for _ in range(10):
                new = {
                    target for source in frontier for value in nodes[source].output
                    for target in consumers[value]
                    if nodes[target].domain in ("", "ai.onnx")
                    and nodes[target].op_type in {"Add", "Mul", "Div", "Pow", "Tanh", "Erf", "Gelu", "Sigmoid", "Reshape", "Identity"}
                    and target not in activation_group
                }
                activation_group.update(new)
                frontier = list(new)
                if not new:
                    break
            if any(nodes[target].op_type in {"Tanh", "Erf", "Gelu", "Sigmoid"} for target in activation_group):
                add("mlp_activation_dag_review", sorted(activation_group), [
                    "Recover the exact activation DAG/constants; Tanh/Erf/Sigmoid alone does not prove GELU/SiLU.",
                    "Do not absorb a following residual or change approximation without proof.",
                ])
            scaled = next_node(index, {"Mul", "Div"})
            if scaled is not None:
                softmax = next_node(scaled, {"Softmax"})
                masked = next_node(scaled, {"Add"})
                if softmax is None and masked is not None:
                    softmax = next_node(masked, {"Softmax"})
                if softmax is not None:
                    value = next_node(softmax, {"MatMul"})
                    if value is not None and nodes[value].input[0] == nodes[softmax].output[0]:
                        group = [index, scaled]
                        if masked is not None and masked != softmax:
                            group.append(masked)
                        add("attention_score_softmax_value", group + [softmax, value], [
                            "Verify Q/K/V layout, exact scale, optional mask semantics and softmax axis.",
                            "Use stable reductions; handle valid tail shapes and all-masked rows if applicable.",
                        ])
        if node.op_type == "Add":
            for norm in consumers[node.output[0]]:
                if nodes[norm].op_type != "LayerNormalization" or nodes[norm].domain not in ("", "ai.onnx"):
                    continue
                kind = "bias_layernorm_review" if any(value in constant_names for value in node.input) else "residual_layernorm"
                add(kind, [index, norm], [
                    "Verify residual broadcasting, normalization axis/epsilon and accumulation dtype.",
                    "Retain any live residual, mean and inverse-standard-deviation outputs.",
                ])
        if node.op_type == "Transpose":
            following = next_node(index, {"Transpose"})
            if following is not None:
                add("transpose_chain", [index, following], [
                    "Compose exact permutations; verify strides and TensorRT consumer formats before elimination.",
                ])
    for indices in projections.values():
        if len(indices) == 3:
            add("shared_input_qkv_projection_review", indices, [
                "Exactly three shared-input constant-weight projections: verify they really are Q/K/V.",
                "Preserve all biases, Gemm attributes, packed-weight orientation, split ordering and head layout.",
                "Measure added split/transpose/reformat cost; do not assume the projections are equal width.",
            ], parallel=True)
    return {
        "node_count": len(nodes), "candidate_count": len(candidates),
        "candidates": candidates,
        "warning": "Topology review only: no rewrite, legality proof, complete Transformer recognition or speedup prediction.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--layers", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(str(args.onnx), load_external_data=False)
    result = analyze(
        model, json.loads(args.layers.read_text())["Layers"],
        json.loads(args.profile.read_text()),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['candidate_count']} fusion review candidates saved to {args.output}")


if __name__ == "__main__":
    main()
