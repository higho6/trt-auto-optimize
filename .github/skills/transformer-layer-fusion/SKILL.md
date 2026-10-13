---
name: transformer-layer-fusion
description: "Use when optimizing Transformer or SigLIP TensorRT graphs: identify QKV, GEMM epilogue, residual LayerNorm, MLP activation, attention/softmax, and layout fusion candidates; transfer measured SM100 optimization principles to SM120 while preserving exact ONNX semantics and validating whole-engine speedup."
---

# Transformer Layer Fusion for TensorRT

## Objective

Reduce Transformer execution time through evidence-backed graph and kernel fusion,
not by merely reducing the number of ONNX nodes. Support the current SigLIP vision
workload and preserve the original ONNX input/output contract.

This is a project skill for the OpenAI workflow, not a Claude plugin.
Use it together with all discovered KDA skills:

- `KernelWiki`: query optimization patterns, trace original sources, and record
  confidence, architecture and version limitations.
- `ncu-report-skill`: profile, diagnose and measure the actual execution kernels.

## Required resources

Read these before proposing an implementation:

- [Architecture transfer](./references/architecture-transfer.md)
- [Fusion patterns and legality](./references/fusion-patterns.md)
- [Topology analysis helper](./scripts/analyze_graph.py)

Use `inspect_transformer_graph` to execute the topology helper on the original
ONNX and baseline TensorRT reports. It produces review candidates, not rewritten
ONNX or proof of fusion legality. Use `query_kernel_wiki` for relevant KDA patterns,
then read cited pages and sources through `read_skill_resource`.

## Procedure

### 1. Establish the contract and baseline

1. Inspect actual GPU properties and the installed CUDA/TensorRT/compiler versions.
2. Validate baseline outputs against original ONNX before creating candidates.
3. Fix the workload: batch, sequence length, hidden width, heads, head dimension,
   precision, layouts, input distribution, and any dynamic-shape range.
4. Rank measured engine hotspots by time. Read layer inputs, outputs, tactic and
   origin metadata. Tactic names do not establish the machine-code target.
5. Inspect the original topology. Distinguish already fused TensorRT execution
   layers from ONNX subgraphs that still incur separate launches or conversions.

### 2. Research and diagnose

1. Query KernelWiki for `epilogue fusion`, `kernel fusion`, `register pressure`,
   `pipeline stages`, `vectorized loads`, or the specific measured kernel.
2. Read the relevant page and its sources. Label the source architecture and
   confidence; a B200 performance claim is not an RTX 5060 Ti performance claim.
3. Use NCU full/PM sampling and source collection, preferably through a standalone
   harness compiled with `-lineinfo` for custom kernels.
4. Follow all six dimensions in the NCU skill: occupancy/geometry, block balance,
   stalls, Tensor Core utilization, timeline, and memory access.
5. Record actual metric values/units and missing metrics. Do not diagnose
   bandwidth-bound behavior from low throughput alone.

### 3. Rank fusion opportunities

Check these groups, in descending order of measured potential benefit:

- GEMM + bias + activation/scale or residual epilogue.
- Residual Add + LayerNorm; preserve pre/post-norm placement and all live outputs.
- Shared-input Q/K/V projections, packed weights, and compatible consumer layouts.
- Attention score GEMM + scale/mask + stable softmax + value GEMM.
- MLP first projection + exact GELU/SiLU, or gated activation if present.
- Redundant transpose/reshape/reformat chains with proven layout equivalence.
- A larger Transformer block only after individual boundaries are understood.

Do not add SwiGLU, causal masks, dropout, or text-model assumptions to SigLIP unless
they actually exist in the graph. Keep SigLIP's attention pooling head separate
from encoder self-attention.

For every proposed group, record node IDs, boundary tensors and shapes, live
consumers, known engine-layer mapping, measured time and blockers. Metadata-based
mapping is only a hint. Do not sum overlapping layer times as expected speedup.

### 4. Prove semantic legality before implementation

- Preserve broadcasting, Gemm alpha/beta/transposes, axes and output ordering.
- Preserve exact GELU approximation and constants; a Tanh/Erf node alone is not
  proof that a subgraph is GELU.
- Preserve LayerNorm epsilon, normalization axis, accumulation dtype, and any
  auxiliary mean/inverse-standard-deviation outputs.
- Preserve mask type, mask application order, attention scale, softmax axis and
  stable reduction. An all-masked row must follow the original behavior.
- Preserve graph-output tensors and intermediate tensors with external consumers.
- Respect strides, tensor packing, dynamic dimensions and plugin output inference.
- If quantization is proposed, use an explicit user-approved accuracy contract
  and calibration/representative data. Do not equate FP4 availability with safe use.

### 5. Implement one candidate at a time

Write `plan.txt` with the measured target, expected mechanism, legality checks,
source references, SM100-to-SM120 classification and rejection criteria.

Prefer exposing a legal pattern to the TensorRT builder before writing a plugin.
If TensorRT already fuses it, do not break that fusion just to replace one node.
For a custom CUDA implementation, create both a standalone correctness/performance
harness and the TensorRT Plugin/ONNX integration. Use the plugin's enqueue stream,
avoid runtime host synchronization/allocation, implement correct dtype/layout and
dynamic-shape handling, and build for the actual target (e.g. `sm_120`).

Use a compiler-supported SM120 backend for architecture-sensitive implementation.
Tune tile dimensions, stages, vector width and registers on the actual workload;
do not copy B200 tile/stage/SM-count constants.

### 6. Profile, evaluate and document

1. Profile the custom harness and the rebuilt engine. Check that the desired
   fusion actually occurred and did not add reformat or spill overhead.
2. Use the independent evaluator against original ONNX. Include edge shapes,
   small/odd sequence sizes and dynamic range boundaries before deployment.
   The default three-seed single-shape gate is necessary, not exhaustive.
3. Measure complete-engine latency under the fixed contract. A faster isolated
   kernel is not sufficient if engine latency regresses.
4. Accept only an evaluator-promoted candidate; do not lower tolerances or gates.
5. Keep rejected candidate evidence. Write a report with the NCU skill's setup,
   six analysis dimensions, cited metrics, source limitations and next actions.

## Stop conditions

Stop and explain if numerical legality is unproven, a required SM120 feature is
unsupported, profiling permissions/source mappings are missing, TensorRT already
has a better fused implementation, or the end-to-end improvement gate fails.
