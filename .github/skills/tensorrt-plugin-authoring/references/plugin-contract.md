# Plugin implementation contract

## IPluginV3 capability checklist

| Interface | Required behavior |
| --- | --- |
| IPluginV3 | Return correct core/build/runtime capability pointers; clone owned state safely |
| IPluginV3OneCore | Stable, unique name/version/namespace |
| IPluginV3OneBuild | Number/type of outputs, symbolic output shape expressions, truthful format combinations, profile configuration, bounded workspace |
| IPluginV3OneRuntime | Runtime shape validation, stream-correct enqueue, context attachment, persistent serialization fields |
| IPluginCreatorV3One | Advertised fields, strict validation, construction in build and runtime phases, matching creator tuple |

Check exact signatures in the installed headers. `noexcept` methods must not leak
exceptions. If allocation can fail, return an explicit error/null and log it.
Own plugin fields and their backing data for as long as TensorRT may access them.
Do not serialize pointers, CUDA handles or process-local addresses.

## CUDA launch

Use `inputs`/`outputs` device pointers and TensorRT's `cudaStream_t`. Respect
runtime shapes rather than a hard-coded batch. Use checked integer arithmetic for
element counts/workspace and handle zero/tail dimensions under the stated contract.
Return nonzero and provide an error message for invalid shapes or CUDA launch errors.
Keep all asynchronous buffers alive until the stream finishes.

## ONNX integration

The custom node replaces a specific verified computation. Match the creator tuple,
field names/types, input ordering and number of outputs. `plugin_namespace` is a
TensorRT lookup attribute, not the same concept as the ONNX domain.

Prefer annotated custom nodes in a candidate graph; standard ONNX shape inference
does not automatically understand plugin semantics. ONNX Runtime generally cannot
execute TensorRT plugin nodes, so evaluate against the unchanged original graph.
Do not compare the candidate only against itself.

If replacing multiple nodes, compute subgraph boundary tensors from producer/
consumer relationships. Retain externally consumed intermediates or add equivalent
plugin outputs; do not remove them just because an adjacent residual can be fused.

## Build and deployment

The bundled CMake example accepts an explicit `CMAKE_CUDA_ARCHITECTURES`, default 120,
and locates installed headers/libraries. It is a starting point, not an ABI promise.
Example after copying the assets to a candidate:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j2
```

Load using `trtexec --dynamicPlugins=/absolute/path/libkda_scale.so` for building
and `--loadEngine` runs. Export the C-linkage `getCreators(int32_t& count)` function
returning a persistent `IPluginCreatorInterface* const*` array, as in the example.
Also export `setLoggerFinder(ILoggerFinder*)` for TensorRT's dynamic loader,
and route error reporting through the supplied logger (the example also prints stderr).
Static `REGISTER_TENSORRT_PLUGIN` registration alone does not satisfy this dynamic
library-loading protocol. In Python use `registry.load_library` before deserialization,
keep the handle alive and verify `registry.get_creator(name, version, namespace)`.
Use `register_plugin` to bind the library to the workflow candidate once compiled.

Changing source, creator fields or kernel ABI requires a new candidate/version and
engine rebuild. A deployment must preserve the library as well as the engine.

## Acceptance matrix

- Kernel and full-plugin correctness against independent reference.
- All graph output names/shapes/types and preserved side consumers.
- Min/opt/max intended dynamic shapes and odd/alignment tails.
- Serialize → fresh process load → execute with the same attributes.
- Multiple execution contexts without shared mutable runtime state.
- Same supplied stream, no hidden synchronization or allocator overhead.
- Actual engine plugin layer present; measured full-engine benefit.
- Original ONNX validation and fixed promotion gate passed.
