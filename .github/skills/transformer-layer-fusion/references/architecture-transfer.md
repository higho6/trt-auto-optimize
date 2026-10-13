# Transferring SM100 experience to SM120

SM100/B200 and SM120/RTX 50-series share useful optimization principles, but they
are not interchangeable instruction targets. A source restricted to `sm_100a`
does not establish `sm_120` support.

## Portable principles: actively reuse, then measure

| SM100 experience | SM120 adaptation | Evidence required |
| --- | --- | --- |
| GEMM epilogue fusion | Keep bias, scale, activation or residual near the GEMM output | Fewer launches/bytes and lower whole-engine latency |
| Avoid intermediate materialization | Keep legal producer/consumer data on chip | Memory traffic, launch count, live-output correctness |
| Vectorized/coalesced access | Select aligned vector widths for actual layouts | Transactions, alignment/tail tests, bandwidth |
| Shared-memory layout/swizzling | Tune bank-conflict avoidance on the target | Shared-memory conflict/stall metrics |
| Register budgeting | Balance live values against occupancy and spills | Registers/thread, occupancy, local-memory traffic |
| Software pipelining/multi-buffering | Sweep stage count rather than copying B200 defaults | Stalls, throughput, registers/shared memory |
| Warp specialization | Reuse producer/consumer role separation where worthwhile | Synchronization correctness and measured utilization |
| Tile scheduling/L2 locality | Tune tile size/order and tail behavior for actual SM count/cache | Waves, cache metrics, PM timeline |
| Persistent or fused kernels | Evaluate when launches/tails dominate | Scheduling/resource cost and engine-level improvement |
| Stable online softmax | Reuse algorithmic reduction/streaming ideas | Exact scale/mask semantics and numerical comparison |

These principles should not be discarded merely because a reference says B200.
Equally, their benefit is a hypothesis until measured on the target.

## Feature-specific mechanisms: verify support separately

- `tcgen05`/TMEM accumulation and `cta_group::2` cooperative MMA from SM100
  references must not be transplanted as SM120 instructions. Preserve the fusion
  or scheduling idea using a supported backend instead.
- TMA, asynchronous copy, mbarrier, cluster launch control, PDL/GDC and low-precision
  formats each need target/toolkit/API support checks for the exact instruction
  variant being used. Do not infer support or non-support from the marketing
  architecture name alone.
- FP8/FP4 block scaling, layout, rounding and supported MMA operand combinations
  are implementation- and version-specific; verify compiler/library dispatch.
- B200 HBM bandwidth, SM count, shared-memory/TMEM capacities and NVLink figures
  are not RTX 5060 Ti resource parameters.

## Required candidate record

For each borrowed technique record:

1. Local KernelWiki page/source and its architecture/confidence/version.
2. The portable algorithmic principle.
3. Any architecture-specific instructions or resource assumptions.
4. The SM120 implementation and compile target, or an explicit unsupported status.
5. Correctness results, actual NCU metrics and complete-engine latency.

Compilation success proves compiler acceptance, not optimal dispatch or speed.
An `sm80` tactic label is not proof of the final machine-code architecture.

## Local KDA sources to consult

Use the `KernelWiki` resource tool to read `references/primer.md`, then
`wiki/techniques/epilogue-fusion.md`, `wiki/techniques/kernel-fusion.md`,
`wiki/techniques/register-budgeting.md` and their cited sources.
Use the `ncu-report-skill` resource tool to read
`blackwell-cuda-programming.md` as SM100-specific background, and
`reference/05-analysis-dimensions.md` for the transferable diagnosis process.
Preserve upstream attribution and license terms when reusing actual code.
