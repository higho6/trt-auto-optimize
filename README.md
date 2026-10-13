# TensorRT 推理自动 Profiling 与优化

## 项目介绍

本仓库旨在利用 Agent 自动分析（Profiling）和优化 TensorRT 计算图的推理性能，减少手动性能分析与调优的工作量。

围绕 TensorRT 推理流程，Agent 的主要目标包括：

- **自动 Profiling**：采集并分析推理过程中的性能数据，定位计算图中的性能瓶颈。
- **计算图优化**：根据分析结果，探索并应用适合的优化策略，提升推理效率。
- **效果验证**：对比优化前后的推理性能，并验证输出正确性，形成“分析—优化—验证”的迭代流程。

项目以推理性能优化为核心，关注推理延迟、吞吐量及 GPU 资源利用情况，目标是在保证输出正确性的前提下，让 TensorRT 推理调优更加自动化。

## 获取仓库

KDA 及其 KernelWiki、NCU skills 以嵌套 Git submodule 引入，需要递归克隆：

```bash
git clone --recurse-submodules https://github.com/higho6/trt-auto-optimize.git
cd trt-auto-optimize
```

如果已经普通克隆，补齐 submodule：

```bash
git submodule update --init --recursive
```

模型 checkpoint、生成的 ONNX/engine/profile、编译产物和 API Key 不包含在仓库中。
使用默认导出流程前，将本地 SigLIP checkpoint 放入
`ckpt/siglip-base-patch16-224/`，或通过 `--model-path` 指定位置。
运行环境需要可用的 NVIDIA GPU、CUDA、TensorRT/trtexec；
Agent 的 kernel profiling/编译还需要 Nsight Compute、nvcc 和 CMake。
Python 依赖见 `requirements.txt`。

## 固定目录与运行记录

源码保留在 `export/`、`engine/` 和 `model/`，所有生成产物统一放在
仓库根目录的 `artifacts/`（已加入 gitignore）。默认路径相对仓库根目录解析，
从其他工作目录调用脚本也不会改变默认输出位置。

```text
artifacts/
  onnx/
    siglip_vision.onnx             # 单独导出的模型
    siglip_vision_b3.onnx          # 整理后的已有模型
    pipeline-<ID>/model.onnx       # 每次一键运行导出的模型
  engines/<ID>/model.engine
  profiles/<ID>/
    profile.json
    layers.json
    times.json
    trtexec.log
  runs/<ID>/
    logs/                         # 一键运行阶段日志
    baseline/engines -> ...        # 指向该次 engine 目录
    baseline/profiles -> ...       # 指向该次 profile 目录
    baseline_validation.json
    agent/                        # 优化记录、候选代码、评测结果
```

同一轮产物共用 ID，默认每轮创建新 ID，不覆盖之前的结果。
`--artifacts-dir` 可以一次指定一键流程的整个输出根目录；通常无需逐项配置路径。
`--output-dir` 只指定运行记录目录，分类产物仍在 artifacts 根目录下按该目录名归档；
记录目录名必须唯一。baseline 中的符号链接不复制大文件，用该 baseline 路径即可
启动后续分析；备份或移动时应保持整个 artifacts 树及链接目标可访问。
已有 `engine/siglip_vision_b3*`、`engine/siglip_dynamic_fp16/` 和旧 runs
已按此结构迁移；历史日志保留原文，其中的旧绝对路径不再代表当前路径。
Agent 也兼容未迁移的旧版平铺 baseline。

## SigLIP 视觉推理示例

运行 `python model/siglip.py`，使用 `/workspace/ckpt/siglip-base-patch16-224` 中的 checkpoint，仅加载视觉模型和图像预处理器，不使用文本分支或 tokenizer。示例需要联网下载测试图片。

对于单张图片，示例输出：

- `last_hidden_state`：形状为 `[1, 196, 768]` 的图像 patch 特征。
- `pooler_output`：形状为 `[1, 768]` 的整图特征，不是图文匹配概率。

## 导出 SigLIP ONNX

安装依赖后运行：

```bash
pip install -r requirements.txt
python export/export_siglip.py --verify
```

默认从本地 `ckpt/siglip-base-patch16-224` 加载视觉分支，导出到
`artifacts/onnx/siglip_vision.onnx`，使用 FP32 和 ONNX opset 17，不包含文本分支或图像预处理。
导出后自动执行 ONNX 结构检查、`onnxsim` 图简化和 shape inference（类型检查、
严格模式及常量数据传播），再次检查后将最终模型保存到同一输出路径。
简化时禁用 Attention 融合，避免引入 `com.microsoft` 算子及额外数值误差，
保留标准 ONNX opset 17。
简化时使用随机输入检查与原始 ONNX 的等价性（动态模式 batch 为 2，
静态模式使用指定的 batch）
（`rtol=1e-3`、`atol=1e-4`），不固定动态 batch 维度；
简化校验或 shape inference 失败时直接报错。
`--verify` 额外使用 ONNX Runtime 的 CPU 执行器验证最终模型与 PyTorch 的
输出一致性（`rtol=1e-3`、`atol=1e-4`）：动态模式验证 batch 为 1 和 2，
静态模式验证指定的 batch。

输入 `pixel_values` 为 `[batch_size, 3, 224, 224]`，应由对应 checkpoint 的
`AutoImageProcessor` 预处理生成。输出为 `last_hidden_state`
（`[batch_size, 196, 768]`）和 `pooler_output`（`[batch_size, 768]`）。
默认仅 batch 维度动态，图像尺寸固定为模型配置中的尺寸。
指定 `--batch-size N` 则导出固定 batch（N 必须为正整数），输入和两个输出的
第 0 维均固定为 N，推理时必须使用该 batch 大小。例如导出 batch 为 4 的模型：

```bash
python export/export_siglip.py --batch-size 4 \
  --output artifacts/onnx/siglip_vision_b4.onnx --verify
```

可以通过 `--model-path` 指定其他本地 SigLIP checkpoint，通过 `--output` 指定输出路径：

```bash
python export/export_siglip.py \
  --model-path /workspace/ckpt/siglip-base-patch16-224 \
  --output artifacts/onnx/siglip_vision.onnx \
  --verify
```

## TensorRT 转换与逐层 Profiling

使用 `engine/build_engine.sh` 调用本机 `trtexec` 构建 engine 并运行推理：

```bash
# 静态 ONNX 不需要指定动态 shape，batch 必须与导出时一致。
bash engine/build_engine.sh artifacts/onnx/siglip_vision_b4.onnx

# 动态 ONNX：指定构建范围和本次 profiling 的实际输入 shape。
bash engine/build_engine.sh artifacts/onnx/siglip_vision.onnx \
  --output-dir artifacts/runs/siglip_fp16_b4 -- \
  --minShapes=pixel_values:1x3x224x224 \
  --optShapes=pixel_values:4x3x224x224 \
  --maxShapes=pixel_values:8x3x224x224 \
  --shapes=pixel_values:4x3x224x224 \
  --fp16
```

可用 `--trtexec /opt/tensorrt/bin/trtexec` 或环境变量 `TRTEXEC` 指定可执行文件。
`--` 后的参数直接传给 `trtexec`，例如 `--noTF32`、`--warmUp=500`、
`--duration=10`、`--iterations=100`。精度、测速时长等默认沿用本机 `trtexec`
设置（通常 FP32 同时允许 TF32；`--fp16` 是允许 FP16，并非强制所有层使用 FP16）。
脚本负责模型及报告输出参数，不允许覆盖这些参数或使用 `--skipInference`。

默认分别写入 `artifacts/engines/build-<ID>/`、`artifacts/profiles/build-<ID>/`
和 `artifacts/runs/build-<ID>/`，每次创建新目录。
单独指定 `--output-dir DIR` 时，文件写入 `DIR/engines/` 和 `DIR/profiles/`；
还可用 `--engine-dir` / `--profile-dir` 指定分类目录，并在 DIR 内创建链接。
显式目录必须不存在，避免覆盖历史证据。单独 ONNX 导出的 `--output` 是固定文件，
重复导出该文件会覆盖它；需要保留每次导出时使用一键脚本。
输出包括：

| 文件 | 用途 |
| --- | --- |
| `engines/model.engine` | 序列化的 TensorRT engine |
| `profiles/profile.json` | 每个执行层的耗时统计，用平均耗时和耗时占比定位瓶颈 |
| `profiles/layers.json` | detailed 层结构，包括输入输出张量、shape、数据类型、格式和可用的 tactic 等信息 |
| `profiles/times.json` | 每次推理的整体时间记录，用于观察延迟分布 |
| `profiles/trtexec.log` | 完整命令和 verbose 构建、融合、测速日志 |

脚本启用 `--profilingVerbosity=detailed` 和 `--separateProfileRun`，
将逐层 profiling 放在单独运行中，避免 profiler 开销污染常规 benchmark；
逐层时间仍包含 profiling 开销，不能直接等同于正常推理延迟。
转换失败或产物缺失时脚本返回非零状态，日志保留用于诊断。

分析时按 `profile.json` 的层名关联 `layers.json`，优先检查高耗时层的
shape、精度、tactic，以及 Reformat/Shuffle 等数据转换开销。根据张量连接和
构建日志检查融合边界，例如 MatMul/Add/激活、归一化或 Attention 相关计算。
报告描述的是 **TensorRT 优化后的执行层**，一个执行层可能包含多个 ONNX 算子；
被融合或消除的原始节点不一定有独立耗时。JSON 本身不会判定哪些层一定能继续融合，
需对照原始 ONNX 图提出优化方案，再重新构建并验证性能和数值正确性。
脚本使用 `trtexec` 的默认测试输入，只测性能，不代替真实输入的数值验证。

## OpenAI 驱动的 TensorRT/KDA 优化 Agent

`engine/optimize_agent.py` 使用 **OpenAI Responses API 的 function calling**，
不依赖 Claude、Claude 插件或 Humanize。`kda/` 保持为只读参考，Agent 按其
“任务契约 → profiling → 诊断 → 候选实现 → 正确性验证 → 性能评测 → 晋升/拒绝”
流程执行；每次运行使用独立目录，记录候选、命令、错误和评测证据。

### 全量 skill 加载与工具适配

Agent 不依赖 Claude 的 skill loader：`engine/skill_registry.py` 递归发现
`kda/skills/`、`.github/skills/` 和存在时的 `.agents/skills/` 中所有 `SKILL.md`，
校验 YAML/name/description、拒绝重名，并记录来源和内容哈希。
**首个 OpenAI 请求前自动加载全部发现的 skill 正文**，不是只在提示词中列出名称。
脚本/参考资料按需读取和执行，避免把整个 KernelWiki 灌入上下文。
后续新增的 skill 会自动发现；必要的 KDA submodule 缺失时明确报错。

```bash
python engine/optimize_agent.py --list-skills
```

当前四个 skill：

| Skill | 实际接入 |
| --- | --- |
| `KernelWiki` | 全文加载；本地 query/get-page 脚本；来源、架构、置信度及版本追溯；参考资料和代码资源读取 |
| `ncu-report-skill` | 全文加载；engine/独立 harness 的 full/PM/source 采集；KDA 原始报告、stall hotspot 和时间线 helper；前后对比及六维报告 |
| `transformer-layer-fusion` | 项目新增 skill；ONNX 拓扑分析、已有融合识别、语义/liveness 检查、SM100→SM120 迁移与端到端验证 |
| `tensorrt-plugin-authoring` | kernel→IPluginV3→creator 注册→ONNX custom node→engine 构建/加载→profiling/独立评测；含可编译 CUDA/CMake 示例 |

`list_skills`、`load_skill`、`list_skill_resources`、`read_skill_resource` 是 OpenAI
function tools；`query_kernel_wiki` 和 `get_kernel_wiki_page` 实际执行 KDA 中的
查询/页面工具。`skills_manifest.json` 记录所有 skill；
`events.jsonl` 的 `skill_loaded` / `skill_used` 记录正文加载及资源/helper 使用。
Skill 是指导流程的知识，不会因加载而自动产生优化收益；实际候选仍必须通过评测。

### 将优化 kernel 接入 TensorRT Plugin

`tensorrt-plugin-authoring` skill 位于
`.github/skills/tensorrt-plugin-authoring/`，包含：

- `SKILL.md` 和 `references/plugin-contract.md`：V3 core/build/runtime/creator 接口、
  字段序列化、动态 shape、dtype/layout、stream/workspace、ONNX 子图替换及部署检查。
- `assets/scale_plugin.cu`、`assets/CMakeLists.txt`：可编译的 FP32 linear
  `IPluginV3` 示例，默认编译 SM120，并保留 `-lineinfo`。
- `scripts/make_example.py`：生成等价原始 ONNX 与带 `KdaScale` custom node 的候选。
  scale 示例仅验证 Plugin 接入，不是 Transformer 优化实现。

Agent 应将资源复制到候选目录，替换为实测优化 kernel，编译 `.so`，然后调用：

```text
register_plugin(
  candidate="candidate_id",
  library="build/liboptimized.so",
  name="OptimizedOperator",
  version="1",
  namespace=""
)
```

注册工具在新进程中加载 `.so`，验证 creator tuple，记录库哈希。
使用 TensorRT `registry.load_library` 的动态加载协议，示例导出
`getCreators` 和 `setLoggerFinder`，并非只用静态注册宏。
之后 `build_candidate` 和 `profile_kernel` 自动传入该库的 `--dynamicPlugins`，
`evaluate_candidate` 的 `plugins: []` 自动使用同一注册列表。
直接透传未注册 Plugin 构建参数会报错；库修改后必须重新注册/新建候选，
不能拿改变过的 `.so` 继续使用原 engine 评测记录。

ONNX custom node 的 `op_type`、`plugin_version`、`plugin_namespace` 和参数字段
必须与 creator 对应；只改候选 ONNX，原图继续用于 ONNX Runtime 数值参考。
部署时保留并在反序列化前加载 `.so`，不要以为 engine 默认包含库代码。
Plugin 必须在 layer 报告和实际执行中出现，且通过完整 engine 的正确性和性能门槛。

### 先准备满足正确性要求的 baseline

默认容差与 ONNX 导出校验一致：`rtol=1e-3`、`atol=1e-4`。TensorRT 的默认
TF32 或 FP16 可能不满足该容差，不能直接把性能更快当作优化成功。
例如先构建禁用 TF32 的 FP32 baseline：

```bash
bash engine/build_engine.sh artifacts/onnx/siglip_vision_b3.onnx \
  --output-dir artifacts/runs/siglip_b3_fp32 -- --noTF32
```

动态 ONNX 仍需传入 `--minShapes`、`--optShapes`、`--maxShapes` 和 `--shapes`。
Agent 的 `--shapes` 则是 **JSON 格式的实际评测输入维度**。第一版评测器支持
普通 FP32 输入、FP32/FP16 输出及 linear/device I/O，不支持 shape-tensor、
整数输入、host I/O 或非线性打包格式，遇到这些情况会明确报错。

### 本地分析，不调用 OpenAI

```bash
python engine/optimize_agent.py --offline \
  --onnx artifacts/onnx/siglip_vision_b3.onnx \
  --baseline artifacts/runs/siglip_b3_fp32 \
  --shapes '{"pixel_values":[3,3,224,224]}'
```

离线模式只关联逐层耗时与结构、排序热点并保存张量连接，不执行生成代码、不调用 API，
也不声称已经优化计算图。

### 启动 OpenAI Agent

#### 一键启动（推荐）

`engine/run_profile.sh` 串联依赖/skills 检查、静态 ONNX 导出（含简化、
shape inference 和数值校验）、TensorRT baseline/profile 构建、baseline 正确性
校验和 OpenAI Agent。默认 batch=3、FP32/noTF32，每次创建独立输出目录，不覆盖旧产物。
不自动安装依赖。默认 Python 为 PATH 中的 `python`，可用 `--python` 指定。

在专用容器/工作机中，将 key 保存到仓库外的私有文件（限制为当前用户可读），运行：

```bash
bash engine/run_profile.sh \
  --api-key-file /private/openai.key \
  --model gpt-4.1 \
  --batch-size 3 \
  --max-turns 30 \
  --max-candidates 3 \
  --allow-api-data \
  --allow-local-execution
```

也可继承环境变量 `OPENAI_API_KEY`，不传 credential 参数。
支持 `--api-key "$OPENAI_API_KEY"`，但 **直接传 key 会暴露到进程 argv，
字面值还可能进入 shell 历史**，因此推荐环境变量或文件。脚本禁用自身 shell tracing，
不打印/记录 key，也不把 key 加入下游命令参数；生成代码仍具备进程环境权限，
此脚本不是秘密隔离或执行沙箱。

兼容后端可加 `--base-url https://your-backend.example/v1`；后端必须支持
Responses/function calling，模型必须是该后端提供的模型。
脚本保留未显式覆盖的 `OPENAI_BASE_URL` 环境变量，官方服务用户应自行确认该变量。

仅执行本地准备和 baseline profiling，不调用 API：

```bash
bash engine/run_profile.sh --prepare-only --batch-size 3
```

复用已有 ONNX 和 baseline（仍会验证数值正确性）：

```bash
bash engine/run_profile.sh \
  --onnx artifacts/onnx/siglip_vision_b3.onnx \
  --shapes '{"pixel_values":[3,3,224,224]}' \
  --baseline artifacts/runs/siglip_b3_fp32 \
  --api-key-file /private/openai.key \
  --allow-api-data --allow-local-execution
```

已有 ONNX 必须显式给 `--shapes`，不要让脚本猜测输入名或维度。
动态 ONNX 构建时用重复的 `--build-arg` 传 min/opt/max/shapes：

```bash
bash engine/run_profile.sh --prepare-only \
  --onnx artifacts/onnx/siglip_vision.onnx \
  --shapes '{"pixel_values":[3,3,224,224]}' \
  --build-arg --minShapes=pixel_values:1x3x224x224 \
  --build-arg --optShapes=pixel_values:3x3x224x224 \
  --build-arg --maxShapes=pixel_values:8x3x224x224 \
  --build-arg --shapes=pixel_values:3x3x224x224
```

构建参数同步传给 Agent 的候选构建；复用 baseline 时必须提供与其构建条件一致的参数。
同名 `--build-arg` 使用最后指定的值覆盖默认值，避免向 trtexec 传重复选项；
默认 noTF32 不自动切换成低精度。
`--ncu-python` 可显式指定，否则从 `ncu` 实际可执行文件所在目录寻找 `extras/python`。
支持 `--rtol`、`--atol`、`--min-speedup`、`--command-timeout` 和工具路径等参数；
全部参数见 `bash engine/run_profile.sh --help`。

默认分类根目录为 `artifacts/`，用 `--artifacts-dir DIR` 可整体调整。
运行记录默认在 `artifacts/runs/pipeline-<唯一标识>/`，也可用 `--output-dir`
指定不存在的新目录。该目录的 `logs/` 保存导出/构建/校验/Agent 的日志，
`baseline_validation.json` 和独立 `agent/` 工作目录。
查看 `agent/events.jsonl` 了解进度，只有 `agent/promoted.json` 才是晋升结果；
流程正常结束不保证得到更快的模型。阶段失败立即返回非零并保留证据，
baseline 校验失败时不会调用 OpenAI。

#### 分阶段启动

将 `OPENAI_API_KEY` 设置到运行环境中，不要写进代码或提交到仓库。
安装 `requirements.txt` 的依赖，并在已经授权上传这些数据的环境中运行：

```bash
python engine/optimize_agent.py \
  --model gpt-4.1 \
  --onnx artifacts/onnx/siglip_vision_b3.onnx \
  --baseline artifacts/runs/siglip_b3_fp32 \
  --shapes '{"pixel_values":[3,3,224,224]}' \
  --build-arg=--noTF32 \
  --ncu-python /opt/nvidia/nsight-compute/2025.3.1/extras/python \
  --allow-api-data \
  --allow-local-execution
```

`--model` 可替换为账户中支持 Responses/function calling 的模型，也可通过
`OPENAI_MODEL` 设置默认值。客户端使用 `OPENAI_API_KEY`，可通过 OpenAI SDK
支持的 `OPENAI_BASE_URL` 配置兼容服务；没有显式 `--allow-api-data` 时禁止请求。
`store=False` 禁用 Responses 的服务端存储选项，**不代表没有数据传输，也不承诺零保留**。
报告摘要、Agent 主动读取的参考内容、候选代码及工具日志会进入模型上下文。
全部发现的 skill 正文也会发送，因此授权前应检查新增 skill 的内容和来源。

**本地执行不是沙箱。** `--allow-local-execution` 授权执行模型生成的代码和命令，
应在独立容器/专用工作机运行，不要在含凭据或其他用户任务的共享主机上开启。
工具限制读写路径和候选状态，但生成的 Python/CUDA/脚本本身具有当前用户权限。
不加该开关时所有本地执行工具明确返回错误。Agent 不自动安装系统依赖。

### Agent 工具与优化闭环

- 检查真实 GPU/SM、CUDA 和 TensorRT 版本；先校验 baseline 的数值正确性。
- 读取详细层信息、热点、张量连接和构建日志；关联原始 ONNX 图。
- `inspect_transformer_graph` 调用新增融合 skill 的拓扑 helper，输出
  `fusion_candidates.json`：GEMM/MLP epilogue、残差 LayerNorm、已有 packed QKV、
  Attention score/softmax/value 及布局链等 review 候选。
  保留 side-consumer/live-output blockers，并将层名/metadata 对应明确标注为提示，
  不自动改图或把候选数当作加速收益。SigLIP 的 pooling head 不等同于 encoder。
- 为每个候选创建 `plan.txt` 和独立目录，生成 ONNX 改写、CUDA kernel、
  TensorRT Plugin、构建脚本或候选构建参数，不修改原始 ONNX/engine 或 `kda/`。
- 候选通过现有 `build_engine.sh` 重建 engine 和逐层报告。
  自定义 kernel 必须通过 ONNX/Plugin 接入实际 TensorRT 图；不是单独跑得快就能晋升。
- NCU 收集 `full` / `source` 报告，读取规则建议，并通过 `engine/ncu_metrics.py`
  和本机 `ncu_report` Python API 枚举实际可用指标，不硬编码 B200 指标。
  `full` 模式检测本机 section 后启用可用的 PM sampling；缺失 section 和不可用
  metric 会明确记录，`metrics_summary.json` 提供概览，完整指标保存在 `metrics.json`。
  `--ncu-python` 应指向当前安装的 Nsight Compute `extras/python`。
  自定义 kernel 应带 `-lineinfo`；TensorRT 内置 kernel 可能没有源代码映射。
  NCU 权限不足、kernel regex 未匹配或分析失败均记录为错误，不伪造 profiling 结论。
- `profile_harness` 支持在候选目录编译的独立 CUDA harness，`profile_kernel`
  支持 baseline/候选 engine。每次 NCU run 按 KDA 布局创建 `harness/`、
  `reports/`、`analysis/` 和 `REPORT.md`。
  采集目录固定为工作目录下的 `profiles/ncu/<stage_ID>/`；候选 TensorRT
  构建产物在 `build/engines/` 与 `build/profiles/`，命令日志在 Agent 的 `logs/`。
  实际调用 KDA 的 `analyze_reports.py`，source 模式调用
  `extract_stall_hotspots.py`，full 模式按实际可用 PM 指标调用 `plot_timeline.py`。
  原始 helper 的 B200 预设只是补充，`metrics.json` 中枚举的实际 GPU 指标是权威数据；
  没有采样/源代码映射时不假装得到了诊断。
- `compare_ncu_reports` 同时生成原生指标对照和 KDA 原始比较报告；
  `write_profile_report` 用于填写 NCU skill 的六维证据分析。
  默认 `REPORT.md` 明确标记“采集完成、诊断待补充”，不能充当性能结论。
- `engine/evaluate_engines.py` 独立校验 baseline 和 candidate：使用 3 组确定种子的
  随机输入，对照 **原始 ONNX 的 ONNX Runtime 输出**，检查所有输出名、shape、有限值
  和固定数值容差。随机输入验证并非真实数据集验证；部署前仍需真实图像回归。
- 正确性通过后，同样输入预热并交替测量两个完整 engine，使用非默认 CUDA stream
  和 CUDA events，默认 5 轮、每轮 100 次，不计数据拷贝、不启用 profiler/CUDA graph。
  只有中位数加速比至少 **1.03** 且至少 **80% 配对轮次更快**才晋升；
  相同 engine 二进制永不因测速噪声晋升。共享 GPU 的干扰仍需人工控制。

可以在启动前用 `--rtol`、`--atol`、`--min-speedup` 定义任务契约；
Agent 无权改变固定评测器中的这些阈值。`--build-arg=...` 必须与 baseline 构建条件一致，
动态 profile 的参数也用该选项传入。`--max-turns`（默认 30）、`--max-candidates`
（默认 3）和 `--command-timeout`（默认 900 秒）限制单次运行预算。
**SM100 经验不是整体排除。** 新增 skill 的 `references/architecture-transfer.md`
明确复用 epilogue/算子融合、减少中间访存、向量化、寄存器预算、软件流水、
warp specialization 和 tile scheduling 等原则，再在 SM120 实测调参。
`tcgen05`/TMEM/2-SM 等 SM100 专属机制不能作为 SM120 指令直接照搬；
其他特性按精确指令/API/工具链验证支持性，不根据“Blackwell”名称一概接受或拒绝。

单独启动 Agent 的默认记录位于 `artifacts/runs/<唯一时间戳>/`：

- `baseline_summary.json`、`baseline_validation.json`：热点和 baseline 校验证据。
- `events.jsonl`：任务契约、输入哈希、模型工具调用、命令日志和候选状态。
- `skills_manifest.json`：所有发现的 skill、来源、描述及内容哈希。
- `fusion_candidates.json`：基于真实 ONNX 拓扑的融合 review 候选与语义阻碍。
- `candidates/<ID>/`：计划、代码、engine、逐层报告、NCU 报告和 `evaluation.json`。
- `promoted.json`：仅由独立评测器通过的候选产生，作为选用候选的指针；
  **不覆盖原有 engine**。未通过则记录拒绝原因。
- `final.txt`：Agent 的总结，不代替评测器的晋升证据。

单元测试（无需 API key，不发送网络请求）：

```bash
python -m unittest discover -s engine -p 'test_*.py'
```
