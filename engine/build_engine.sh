#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash engine/build_engine.sh MODEL.onnx [--output-dir DIR] [--engine-dir DIR] [--profile-dir DIR] [--trtexec PATH] [-- TRTEXEC_ARGS...]

Build a TensorRT engine and export detailed layer information and timings.
Defaults: artifacts/{engines,profiles,runs}/build-<unique ID>/.
With --output-dir, use DIR/engines/ and DIR/profiles/ unless explicitly overridden.
Explicit output directories must be new; existing runs are never overwritten.

Examples:
  bash engine/build_engine.sh artifacts/onnx/siglip_vision_b3.onnx
  bash engine/build_engine.sh artifacts/onnx/siglip_vision.onnx -- \
    --minShapes=pixel_values:1x3x224x224 --optShapes=pixel_values:4x3x224x224 \
    --maxShapes=pixel_values:8x3x224x224 --shapes=pixel_values:4x3x224x224 --fp16

Output: engines/model.engine; profiles/{profile.json,layers.json,times.json,trtexec.log}
Separate external directories are linked from the run directory for baseline reuse.
EOF
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi
if [[ $1 == --help || $1 == -h ]]; then
    usage
    exit 0
fi
onnx_path=$1
shift
[[ -f $onnx_path ]] || fail "ONNX file not found: $onnx_path"
onnx_path=$(realpath -- "$onnx_path")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_dir=
engine_dir=
profile_dir=
trtexec_bin=${TRTEXEC:-trtexec}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir|--engine-dir|--profile-dir|--trtexec)
            [[ $# -ge 2 && -n $2 && $2 != --* ]] || fail "Missing value for $1"
            case "$1" in
                --output-dir) output_dir=$2 ;;
                --engine-dir) engine_dir=$2 ;;
                --profile-dir) profile_dir=$2 ;;
                --trtexec) trtexec_bin=$2 ;;
            esac
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            fail "Unknown script option: $1 (put trtexec options after --)"
            ;;
    esac
done

for arg in "$@"; do
    case "${arg%%=*}" in
        --onnx|--saveEngine|--loadEngine|--skipInference|--help|--profilingVerbosity|--exportProfile|--exportLayerInfo|--exportTimes)
            fail "Option is managed by this script and cannot be overridden: $arg"
            ;;
    esac
done
command -v "$trtexec_bin" >/dev/null 2>&1 || fail "trtexec not found: $trtexec_bin (use --trtexec PATH or TRTEXEC)"
if [[ -z $output_dir ]]; then
    artifacts_dir="$script_dir/../artifacts"
    mkdir -p -- "$artifacts_dir/runs"
    output_dir=$(mktemp -d "$artifacts_dir/runs/build-XXXXXXXX")
    run_id=$(basename -- "$output_dir")
    engine_dir=${engine_dir:-"$artifacts_dir/engines/$run_id"}
    profile_dir=${profile_dir:-"$artifacts_dir/profiles/$run_id"}
else
    [[ ! -e $output_dir && ! -L $output_dir ]] || fail "Output directory already exists: $output_dir"
    mkdir -p -- "$output_dir"
fi
output_dir=$(realpath -- "$output_dir")
for kind in engines profiles; do
    if [[ $kind == engines ]]; then directory=${engine_dir:-"$output_dir/engines"};
    else directory=${profile_dir:-"$output_dir/profiles"}; fi
    [[ ! -e $directory && ! -L $directory ]] || fail "Artifact directory already exists: $directory"
    mkdir -p -- "$directory"
    directory=$(realpath -- "$directory")
    if [[ $directory != "$output_dir/$kind" ]]; then
        ln -sr -- "$directory" "$output_dir/$kind"
    fi
    if [[ $kind == engines ]]; then engine_dir=$directory; else profile_dir=$directory; fi
done

command=(
    "$trtexec_bin"
    "--onnx=$onnx_path"
    "--saveEngine=$engine_dir/model.engine"
    --profilingVerbosity=detailed
    --dumpLayerInfo
    --dumpProfile
    --separateProfileRun
    "--exportProfile=$profile_dir/profile.json"
    "--exportLayerInfo=$profile_dir/layers.json"
    "--exportTimes=$profile_dir/times.json"
    --verbose
    "$@"
)
{
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    "${command[@]}"
} 2>&1 | tee "$profile_dir/trtexec.log"

[[ -s "$engine_dir/model.engine" ]] || fail "trtexec did not produce model.engine; see $profile_dir/trtexec.log"
for artifact in profile.json layers.json times.json; do
    [[ -s "$profile_dir/$artifact" ]] || fail "trtexec did not produce a non-empty $artifact; see $profile_dir/trtexec.log"
done
printf '\nEngine and profiling artifacts saved to: %s\n' "$output_dir"
