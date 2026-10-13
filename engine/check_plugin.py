"""Verify a plugin library registers the exact creator tuple in a fresh process."""

import argparse
import json
from pathlib import Path

import tensorrt as trt


LIBRARIES: dict[Path, object] = {}


def load_plugin_library(path: Path):
    path = path.resolve()
    if path not in LIBRARIES:
        handle = trt.get_plugin_registry().load_library(str(path))
        if handle is None:
            raise RuntimeError(f"TensorRT could not dynamically load plugin library: {path}")
        LIBRARIES[path] = handle
    return LIBRARIES[path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--namespace", default="")
    args = parser.parse_args()
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    load_plugin_library(args.library)
    creator = trt.get_plugin_registry().get_creator(args.name, args.version, args.namespace)
    if creator is None:
        raise RuntimeError(f"Plugin creator not registered: {(args.name, args.version, args.namespace)}")
    print(json.dumps({
        "library": str(args.library.resolve()), "name": creator.name,
        "version": creator.plugin_version, "namespace": creator.plugin_namespace,
    }))


if __name__ == "__main__":
    main()
