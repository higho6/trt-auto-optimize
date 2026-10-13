---
name: tensorrt-plugin-authoring
description: "Use when turning an optimized CUDA kernel or fused Transformer subgraph into a TensorRT Plugin: implement IPluginV3 interfaces, creator registration, dynamic shapes, precision/layout negotiation, serialization, ONNX replacement, shared-library loading, profiling and engine-level correctness/performance validation on SM120 or other verified targets."
---

# TensorRT Plugin Authoring

## Goal

Deliver an optimized kernel as a loadable TensorRT operator that actually executes
inside the rebuilt engine. A standalone CUDA microbenchmark, a compiled `.so`,
or an ONNX custom node alone is not a completed integration.

Use alongside `KernelWiki`, `ncu-report-skill` and `transformer-layer-fusion`.
Keep the original ONNX as the independent correctness reference.

## Resources

- [Implementation and integration checklist](./references/plugin-contract.md)
- [Buildable IPluginV3 scale example](./assets/scale_plugin.cu)
- [Build configuration](./assets/CMakeLists.txt)
- [Reference/custom-node ONNX example](./scripts/make_example.py)

The scale example is a plumbing test, not a proposed Transformer optimization.
Copy assets into a candidate workspace before building. Replace the example
operation with the measured, validated kernel; do not relabel scale as fusion.

## Workflow

1. Validate baseline; inspect installed TensorRT headers, CUDA version and real GPU.
   New TensorRT 10 plugins should use `IPluginV3` / `IPluginCreatorV3One`. Do not
   paste V2 or a different TensorRT release's signatures without checking headers.
2. Use Transformer fusion and KernelWiki skills to define a legal subgraph.
   Record inputs/outputs, attributes, live side outputs, shape ranges, dtypes,
   layouts, normalization/mask semantics and exact numeric contract in `plan.txt`.
3. Validate the CUDA kernel in a standalone harness. Compile for the actual GPU,
   such as `sm_120`, with `-lineinfo`. Use full/PM/source NCU evidence.
4. Implement all plugin capabilities and creator fields following the checklist.
   Invoke the optimized kernel in `enqueue` on TensorRT's supplied stream.
   Report CUDA errors; never silently substitute the reference operation.
5. Compile a shared library linked against the installed TensorRT and CUDA.
   Use a unique creator name/version/namespace for the candidate implementation.
6. Call `register_plugin` with the candidate-relative `.so` and exact creator
   tuple. The tool loads it in a subprocess, verifies creator registration and
   records its hash. It does not prove kernel correctness or graph integration.
7. Rewrite only the candidate ONNX: replace the validated subgraph with a custom
   node whose `op_type` equals the creator name, and whose `plugin_version` /
   `plugin_namespace` match registration. Supply creator fields as ONNX attributes.
   Remove only dead nodes; preserve output tensor names and live consumers.
   Keep required custom-domain opset and output shape/type annotations.
8. Call `build_candidate`. Registered libraries are automatically passed as
   `--dynamicPlugins` at build time. Check `layers.json`, origin metadata and
   logs to confirm the intended plugin is in the engine, not unused or eliminated.
9. Call `profile_kernel` on the candidate engine. Registered libraries are loaded
   again in the profiling process. Compare the isolated and integrated kernels;
   check reformat overhead and any fusion lost by replacing the original subgraph.
10. Call `evaluate_candidate` with `plugins: []` to use the registered libraries
    automatically. The original ONNX, not the plugin ONNX, supplies expected outputs.
    Test serialize/reload in a fresh process, shape boundaries, multiple contexts,
    graph-capture compatibility where needed, and real inputs before deployment.
11. Only accept independently promoted candidates. Retain the `.so`, creator tuple,
    source/build files, target architectures, TensorRT/CUDA ABI versions, engine,
    hashes, profiling and accuracy evidence together for deployment.

## Failure and portability rules

- No allocation, compilation, device-wide sync or host copies in `enqueue`.
- Do not assume plugins are serialized as library code by default. Deploy/load the
  `.so` before deserialization unless a verified alternative packaging is used.
- Do not allow a changed `.so` to reuse an already built engine/evaluation record.
- Support only dtypes/layouts the kernel implements; unsupported configurations
  must fail explicitly, not be accepted by `supportsFormatCombination`.
- Dynamic batch and shape support require correct symbolic output dimensions,
  runtime shape validation and bounded workspace arithmetic.
- SM100 fusion/pipelining principles transfer; SM100-only instructions do not.
  Use the fusion skill's architecture-transfer guide and verify the backend.
- Source-level profiling may be unavailable for stock TensorRT kernels; custom
  plugin code should retain `-lineinfo`.
- Replacing an already fast TensorRT fusion with a plugin can make the engine slower.
  A successful build is not a successful optimization.
