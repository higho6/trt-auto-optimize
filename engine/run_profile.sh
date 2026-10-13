#!/usr/bin/env bash
# Disable tracing before handling credentials, including when invoked with bash -x.
set +x
set -euo pipefail
umask 077

usage() {
    cat <<'EOF'
Usage: bash engine/run_profile.sh [OPTIONS]

One-shot SigLIP export -> TensorRT baseline/profile -> correctness -> OpenAI agent.
Default: static batch=3, FP32/noTF32; fixed artifacts/{onnx,engines,profiles,runs}.

Credentials/backend:
  --api-key KEY            Set OPENAI_API_KEY (visible in process argv/history).
  --api-key-file FILE      Read key from a private text file (recommended).
                          Otherwise inherit OPENAI_API_KEY from environment.
  --base-url URL           Set OPENAI_BASE_URL for a Responses-compatible backend.
  --model NAME            Model (default OPENAI_MODEL or gpt-4.1).

Inputs/output:
  --batch-size N          Static SigLIP export batch (default 3).
  --model-path DIR        Local checkpoint (default ckpt/siglip-base-patch16-224).
  --onnx FILE             Use existing ONNX instead of exporting; requires --shapes.
  --shapes JSON           Actual input shapes, e.g. '{"pixel_values":[3,3,224,224]}'.
  --baseline DIR          Reuse baseline reports/engine; still validate correctness.
  --artifacts-dir DIR     Artifact root (default: repository artifacts/).
  --output-dir DIR        New run directory; must not already exist.
  --build-arg ARG         Repeatable trtexec argument, e.g. --build-arg --fp16.
                          Defaults include --noTF32, --warmUp=500, --duration=5.
                          Dynamic ONNX requires profile args via --build-arg.

Runtime:
  --python PATH           Interpreter (default python).
  --trtexec PATH          TensorRT tool (default TRTEXEC or trtexec).
  --ncu PATH              Nsight Compute tool (default ncu).
  --ncu-python DIR        Nsight Compute extras/python; auto-detected if omitted.
  --max-turns N           Agent turn budget (default 30).
  --max-candidates N      Candidate budget (default 3).
  --command-timeout N     Per-command timeout seconds (default 900).
  --rtol VALUE            Accuracy relative tolerance (default 1e-3).
  --atol VALUE            Accuracy absolute tolerance (default 1e-4).
  --min-speedup VALUE     Promotion threshold (default 1.03).
  --prepare-only          Export/build/validate/analyze locally; no API calls.
  --allow-api-data        Consent to sending reports/code/skills/logs to backend.
  --allow-local-execution Consent to executing generated code (NOT sandboxed).
  --help                 Show help.

Example:
  bash engine/run_profile.sh --api-key-file /private/openai.key --model gpt-4.1 \
    --batch-size 3 --allow-api-data --allow-local-execution

Run generated-code execution only in a dedicated container/host.
EOF
}

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=python
trtexec_bin=${TRTEXEC:-trtexec}
ncu_bin=ncu
ncu_python=
model=${OPENAI_MODEL:-gpt-4.1}
checkpoint="$root/ckpt/siglip-base-patch16-224"
batch=3
onnx=
shapes=
baseline=
output=
artifacts_dir="$root/artifacts"
key_file=
key_argument=
key_source=
prepare=false
allow_api=false
allow_execution=false
max_turns=30
max_candidates=3
command_timeout=900
rtol=1e-3
atol=1e-4
min_speedup=1.03
build_args=(--noTF32 --warmUp=500 --duration=5)

while [[ $# -gt 0 ]]; do
    option=$1
    case "$option" in
        --help|-h) usage; exit 0 ;;
        --prepare-only) prepare=true; shift; continue ;;
        --allow-api-data) allow_api=true; shift; continue ;;
        --allow-local-execution) allow_execution=true; shift; continue ;;
        --api-key|--api-key-file|--base-url|--model|--batch-size|--model-path|--onnx|--shapes|--baseline|--output-dir|--artifacts-dir|--build-arg|--python|--trtexec|--ncu|--ncu-python|--max-turns|--max-candidates|--command-timeout|--rtol|--atol|--min-speedup)
            [[ $# -ge 2 && -n $2 ]] || fail "Missing value for $option"
            if [[ $option != --build-arg && $2 == --* ]]; then
                fail "Missing value for $option"
            fi
            ;;
        *) fail "Unknown option: $option (use --help)" ;;
    esac
    case "$option" in
        --api-key|--api-key-file)
            [[ -z $key_source ]] || fail "Specify only one credential argument"
            key_source=$option
            if [[ $option == --api-key ]]; then key_argument=$2; else key_file=$2; fi ;;
        --base-url) export OPENAI_BASE_URL=$2 ;;
        --model) model=$2 ;;
        --batch-size) batch=$2 ;;
        --model-path) checkpoint=$2 ;;
        --onnx) onnx=$2 ;;
        --shapes) shapes=$2 ;;
        --baseline) baseline=$2 ;;
        --output-dir) output=$2 ;;
        --artifacts-dir) artifacts_dir=$2 ;;
        --build-arg)
            filtered=()
            for arg in "${build_args[@]}"; do
                if [[ ${arg%%=*} != "${2%%=*}" ]]; then filtered+=("$arg"); fi
            done
            build_args=("${filtered[@]}" "$2")
            ;;
        --python) python_bin=$2 ;;
        --trtexec) trtexec_bin=$2 ;;
        --ncu) ncu_bin=$2 ;;
        --ncu-python) ncu_python=$2 ;;
        --max-turns) max_turns=$2 ;;
        --max-candidates) max_candidates=$2 ;;
        --command-timeout) command_timeout=$2 ;;
        --rtol) rtol=$2 ;;
        --atol) atol=$2 ;;
        --min-speedup) min_speedup=$2 ;;
    esac
    shift 2
done
for value in "$batch" "$max_turns" "$max_candidates" "$command_timeout"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || fail "Batch size and runtime budgets must be positive integers"
done
command -v "$python_bin" >/dev/null || fail "Python interpreter not found"
command -v "$trtexec_bin" >/dev/null || fail "trtexec not found"
if [[ -n $onnx ]]; then
    [[ -f $onnx && -n $shapes ]] || fail "--onnx requires an existing file and explicit --shapes JSON"
    onnx=$(realpath -- "$onnx")
else
    [[ -d $checkpoint ]] || fail "Checkpoint directory not found"
    checkpoint=$(realpath -- "$checkpoint")
    [[ -z $shapes ]] || fail "--shapes is only used with an existing --onnx; export uses --batch-size"
    shapes=$("$python_bin" -c '
import json, sys
config = json.load(open(sys.argv[1]))
size = config.get("vision_config", config).get("image_size", 224)
print(json.dumps({"pixel_values": [int(sys.argv[2]), 3, size, size]}))
' "$checkpoint/config.json" "$batch")
fi
[[ -z $baseline || -d $baseline ]] || fail "Baseline directory not found"
if [[ -n $baseline ]]; then baseline=$(realpath -- "$baseline"); fi
"$python_bin" -c '
import json, math, sys
shapes = json.loads(sys.argv[1])
if not isinstance(shapes, dict) or not shapes or any(
    not isinstance(name, str) or not isinstance(shape, list) or not shape
    or any(type(dim) is not int or dim <= 0 for dim in shape)
    for name, shape in shapes.items()
): raise ValueError("Invalid shapes JSON")
rtol, atol, speedup = map(float, sys.argv[2:])
if not all(map(math.isfinite, (rtol, atol, speedup))) or min(rtol, atol) < 0 or speedup <= 1:
    raise ValueError("Invalid tolerances or speedup threshold")
' "$shapes" "$rtol" "$atol" "$min_speedup"

if [[ $prepare == false ]]; then
    [[ $allow_api == true && $allow_execution == true ]] || fail "Agent mode requires --allow-api-data and --allow-local-execution (not sandboxed)"
    if [[ -n $key_file ]]; then
        [[ -f $key_file && -r $key_file ]] || fail "API key file is not readable"
        OPENAI_API_KEY=$(< "$key_file")
        export OPENAI_API_KEY
    elif [[ -n $key_argument ]]; then
        export OPENAI_API_KEY=$key_argument
        printf 'Warning: --api-key exposes the key in argv/history; prefer environment or --api-key-file.\n' >&2
    fi
    unset key_argument
    [[ -n ${OPENAI_API_KEY:-} ]] || fail "Set OPENAI_API_KEY or supply --api-key-file/--api-key"
    command -v "$ncu_bin" >/dev/null || fail "ncu not found"
    for tool in nvcc cmake; do command -v "$tool" >/dev/null || fail "$tool not found"; done
    if [[ -z $ncu_python ]]; then
        resolved_ncu=$(realpath -- "$(command -v "$ncu_bin")")
        ncu_python="$(dirname -- "$resolved_ncu")/extras/python"
    fi
    [[ -f $ncu_python/ncu_report.py ]] || fail "Cannot locate ncu_report.py; set --ncu-python"
    ncu_python=$(realpath -- "$ncu_python")
fi
"$python_bin" -c 'import onnx, onnxruntime, torch, tensorrt, yaml'
if [[ -z $onnx ]]; then "$python_bin" -c 'import transformers, onnxsim'; fi
if [[ $prepare == false ]]; then "$python_bin" -c 'import openai'; fi
"$python_bin" "$root/engine/optimize_agent.py" --list-skills

if [[ -z $output ]]; then
    mkdir -p -- "$artifacts_dir/runs"
    output=$(mktemp -d "$artifacts_dir/runs/pipeline-XXXXXXXX")
else
    [[ ! -e $output && ! -L $output ]] || fail "Output directory already exists; use a new directory"
    mkdir -p -- "$output"
fi
output=$(realpath -- "$output")
mkdir -p -- "$artifacts_dir"
artifacts_dir=$(realpath -- "$artifacts_dir")
run_id=$(basename -- "$output")
onnx_dir="$artifacts_dir/onnx/$run_id"
engine_dir="$artifacts_dir/engines/$run_id"
profile_dir="$artifacts_dir/profiles/$run_id"
for directory in "$onnx_dir" "$engine_dir" "$profile_dir"; do
    [[ ! -e $directory && ! -L $directory ]] || fail "Run artifact directory already exists: $directory"
done
mkdir -p -- "$output/logs"
printf 'Pipeline output: %s\n' "$output"
trap 'status=$?; if (( status != 0 )); then printf "Pipeline failed (exit %s). Evidence retained in %s\n" "$status" "$output" >&2; fi' EXIT

if [[ -z $onnx ]]; then
    mkdir -p -- "$onnx_dir"
    onnx="$onnx_dir/model.onnx"
    printf '\n[1/4] Exporting and verifying static ONNX\n'
    "$python_bin" -u "$root/export/export_siglip.py" --model-path "$checkpoint" \
        --batch-size "$batch" --output "$onnx" --verify 2>&1 | tee "$output/logs/export.log"
fi
if [[ -z $baseline ]]; then
    baseline="$output/baseline"
    printf '\n[2/4] Building TensorRT baseline and layer profiles\n'
    bash "$root/engine/build_engine.sh" "$onnx" --output-dir "$baseline" \
        --engine-dir "$engine_dir" --profile-dir "$profile_dir" \
        --trtexec "$trtexec_bin" -- "${build_args[@]}" 2>&1 | tee "$output/logs/build.log"
fi
baseline_engine="$baseline/engines/model.engine"
baseline_profiles="$baseline/profiles"
if [[ ! -d $baseline/engines ]]; then baseline_engine="$baseline/model.engine"; fi
if [[ ! -d $baseline_profiles ]]; then baseline_profiles=$baseline; fi
[[ -s $baseline_engine ]] || fail "Baseline engine missing: $baseline_engine"
for artifact in layers.json profile.json times.json; do
    [[ -s $baseline_profiles/$artifact ]] || fail "Baseline artifact missing: $artifact"
done
printf '\n[3/4] Validating baseline against original ONNX\n'
"$python_bin" -u "$root/engine/evaluate_engines.py" \
    --onnx "$onnx" --baseline "$baseline_engine" --candidate "$baseline_engine" \
    --shapes "$shapes" --rtol "$rtol" --atol "$atol" --min-speedup "$min_speedup" \
    --iterations 10 --repeats 3 --output "$output/baseline_validation.json" \
    2>&1 | tee "$output/logs/validation.log"

agent_args=(
    --onnx "$onnx" --baseline "$baseline" --shapes "$shapes"
    --run-dir "$output/agent" --model "$model" --trtexec "$trtexec_bin"
    --max-turns "$max_turns" --max-candidates "$max_candidates"
    --command-timeout "$command_timeout" --rtol "$rtol" --atol "$atol"
    --min-speedup "$min_speedup"
)
for arg in "${build_args[@]}"; do agent_args+=("--build-arg=$arg"); done
if [[ $prepare == true ]]; then
    printf '\n[4/4] Local report analysis (no API request)\n'
    agent_args+=(--offline)
else
    printf '\n[4/4] Starting OpenAI optimization agent\n'
    agent_args+=(--ncu "$ncu_bin" --ncu-python "$ncu_python" --allow-api-data --allow-local-execution)
fi
"$python_bin" -u "$root/engine/optimize_agent.py" "${agent_args[@]}" 2>&1 | tee "$output/logs/agent.log"
printf '\nPipeline completed. Results: %s\n' "$output"
printf 'An agent summary is not proof of speedup; inspect agent/promoted.json and candidate evaluation.json.\n'
