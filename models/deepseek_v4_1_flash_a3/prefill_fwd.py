# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: manual
"""Run the complete Prefill composition with caller-owned weights, cache metadata and outputs."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import argparse
import json
from pathlib import Path


def run_prefill(inputs, *, weights, cache, config, devices, output, continuation=None):
    """Execute the existing network using explicit checkpoint and cache-state providers.

    ``inputs`` contains the backbone input tensors and input contract.
    ``cache`` supplies the canonical all-rank cache ABI, source descriptors and
    Engram contracts. No historical task release or layer receipt is required.
    ``continuation`` accepts one prepared chunk or a lazy iterable of chunks;
    each chunk follows the preceding input's hash and next epoch.
    """
    from .runtime_config import configure_devices

    configure_devices(devices, world=config["topology"]["ep"])
    from .prefill_network_runtime import run_network_runtime

    overlap = set(inputs) & set(cache)
    if overlap:
        raise ValueError("Input and cache configuration have duplicate fields: " + ", ".join(sorted(overlap)))
    return run_network_runtime(
        dict(inputs, **cache), contract=config, checkpoint=weights,
        devices=devices, output=output, continuation=continuation,
    )


def load_configuration(path):
    """Read paths relative to a JSON configuration without importing the device runtime."""
    path = Path(path).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    required = {"checkpoint", "inputs", "cache_state", "contract", "devices", "output"}
    missing = required - settings.keys()
    if missing:
        raise ValueError("Missing Prefill configuration fields: " + ", ".join(sorted(missing)))
    for name in ("checkpoint", "inputs", "cache_state", "contract", "output"):
        value = Path(settings[name])
        settings[name] = value if value.is_absolute() else (path.parent / value).resolve()
    chunks = settings.get("continuations", [])
    if not isinstance(chunks, list):
        raise ValueError("continuations must be a list of prepared chunk descriptors")
    for chunk in chunks:
        if not isinstance(chunk, dict) or not {"inputs", "contract", "metadata_sources"} <= chunk.keys():
            raise ValueError("Each continuation needs inputs, contract and metadata_sources paths")
        for name in ("inputs", "contract", "metadata_sources"):
            if not isinstance(chunk[name], str) or not chunk[name]:
                raise ValueError("Continuation paths must be nonempty strings")
            value = Path(chunk[name])
            chunk[name] = value if value.is_absolute() else (path.parent / value).resolve()
    devices = settings["devices"]
    if (not isinstance(devices, list) or len(devices) != 16 or
            any(type(value) is not int or value < 0 for value in devices) or len(set(devices)) != 16):
        raise ValueError("The current full-network configuration requires 16 distinct physical device IDs")
    return settings


def load_continuations(chunks):
    """Load each prepared chunk and its rank metadata descriptors on demand."""
    import torch

    for chunk in chunks:
        metadata_path = Path(chunk["metadata_sources"])
        sources = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(sources, list) or len(sources) != 16:
            raise ValueError("Continuation metadata_sources must describe all 16 ranks")
        for rank, source in enumerate(sources):
            if not isinstance(source, dict) or not {"rank", "path", "sha256"} <= source.keys():
                raise ValueError("Each continuation metadata source needs rank, path and sha256")
            if type(source["rank"]) is not int or source["rank"] != rank:
                raise ValueError("Continuation metadata sources must retain rank order")
            value = Path(source["path"])
            source["path"] = str(value if value.is_absolute() else (metadata_path.parent / value).resolve())
        inputs = torch.load(chunk["inputs"], map_location="cpu", weights_only=True)
        contract = json.loads(Path(chunk["contract"]).read_text(encoding="utf-8"))
        yield dict(inputs=inputs, contract=contract, metadata_sources=sources)


def main():
    """Load an explicit full-network configuration and run the shared product path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--devices", help="Comma-separated physical device IDs; overrides the JSON mapping")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-config", action="store_true", help="Validate the configuration without device execution")
    args = parser.parse_args()
    settings = load_configuration(args.config)
    if args.devices is not None:
        settings["devices"] = [int(value) for value in args.devices.split(",")]
    if args.checkpoint is not None:
        settings["checkpoint"] = args.checkpoint.resolve()
    if args.output is not None:
        settings["output"] = args.output.resolve()
    # Bind device visibility before loading the compiler, Torch or CANN.
    from .runtime_config import configure_devices, validate_devices

    if args.check_config:
        validate_devices(settings["devices"], world=16)
        print(json.dumps({"configuration": "valid", "execution": "not_run", "platform": args.platform}))
        return
    configure_devices(settings["devices"], world=16)
    import torch

    contract = json.loads(settings["contract"].read_text(encoding="utf-8"))
    inputs = torch.load(settings["inputs"], map_location="cpu", weights_only=True)
    cache = torch.load(settings["cache_state"], map_location="cpu", weights_only=True)
    continuation = load_continuations(settings["continuations"]) if settings.get("continuations") else None
    run_prefill(inputs, weights=settings["checkpoint"], cache=cache, config=contract,
                devices=settings["devices"], output=settings["output"], continuation=continuation)


if __name__ == "__main__":
    main()
