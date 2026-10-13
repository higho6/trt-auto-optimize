# Transformer fusion patterns and legality checks

All patterns are review candidates, not permission to rewrite automatically.

| Pattern | Potential benefit | Critical checks |
| --- | --- | --- |
| MatMul/Gemm + bias + activation | GEMM epilogue; avoid a round-trip and launch | Bias broadcast, transpose/alpha/beta, exact activation, external consumers |
| Output projection + residual | Fuse residual read/add into epilogue | Residual shape/stride, accumulation dtype, pre/post-norm semantics |
| Residual Add + LayerNorm | Reduce reads/writes and launches | Epsilon, axis, gamma/beta, live residual and auxiliary LN outputs |
| Three shared-input Q/K/V projections | Read input once; larger packed GEMM | Weight orientation, all biases, projection widths, split/head ordering |
| QK GEMM + scale/mask + softmax + PV GEMM | Stream attention; avoid score materialization | Stable softmax, mask/order, head layout, arbitrary sequence length |
| MLP projection + decomposed GELU | Fold activation into GEMM epilogue | Recover exact coefficient/DAG and approximation before replacement |
| Consecutive layout operations | Remove unnecessary shuffles/reformats | Strides vs materialization, consumer format, transpose composition |
| Entire Transformer block | Fewer intermediate transfers and launches | Register/shared-memory pressure, occupancy, spills, residual/output liveness |

## Why larger fusion can lose

Two GEMMs with different best tile shapes may not share an efficient schedule.
Keeping too many intermediates live can spill registers, reduce occupancy or
force smaller tiles. Packed QKV may add split/transpose/reformat costs. A plugin
can prevent TensorRT from selecting superior library tactics or other fusions.

Measure both the removed overhead and the introduced overhead. Use a microbench
for diagnosis and the fixed engine evaluator for the promotion decision.

## Suggested correctness matrix

- Current deployment shape and representative real preprocessed images.
- Small batches; each intended dynamic batch minimum/optimum/maximum.
- Sequence/head dimensions with partial tiles and valid non-aligned tails.
- Large/small activations, zero input, scale/mask edge cases when applicable.
- Every graph output and every retained side consumer.
- FP32 reference; candidate precision under the user-defined tolerance.

Do not claim support for an untested dynamic range or shape.
