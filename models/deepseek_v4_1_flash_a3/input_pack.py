# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Token embedding, input packing and initial HC computations."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"
import pypto.language as pl


# tiling
D_TILE = 512


def make_vocab_embedding(capacity, hidden, vocabulary, tp):
    """Copy owned vocabulary rows and zero other active rows before TP reduction."""
    if capacity < 1 or hidden % D_TILE or tp not in (1, 2, 4) or vocabulary % tp:
        raise ValueError("Require aligned hidden width and complete TP1/2/4 vocabulary shards")
    local_rows = vocabulary // tp

    @pl.jit
    def embedding(
        token_ids: pl.Tensor[[capacity], pl.INT64],
        live: pl.Tensor[[capacity], pl.INT32],
        weight: pl.Tensor[[local_rows, hidden], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        vocab_start: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * (hidden // D_TILE), name_hint="prefill_vocab_embedding"):
            token = block // (hidden // D_TILE)
            column = block % (hidden // D_TILE) * D_TILE
            global_id = pl.cast(pl.read(token_ids, [token]), pl.INT32)
            local_id = global_id - vocab_start
            tile = pl.full([1, D_TILE], dtype=pl.BF16, value=0.0)
            if pl.read(live, [token]) > 0:
                if local_id >= 0:
                    if local_id < local_rows:
                        row = pl.cast(local_id, pl.INDEX)
                        tile = weight[row : row + 1, column : column + D_TILE]
            output[token : token + 1, column : column + D_TILE] = tile
        return output

    return embedding


def make_initial_hc(capacity, hidden, hc_mult):
    """Shard tokens, replicate BF16 hidden lanes and set the initial one-hot pre-mix."""
    if capacity < 1 or hidden % D_TILE or hc_mult < 1:
        raise ValueError("Require aligned hidden width and positive HC width")

    @pl.jit
    def initial_hc(
        embedding: pl.Tensor[[capacity, hidden], pl.BF16],
        streams: pl.InOut[pl.Tensor[[capacity, hc_mult, hidden], pl.BF16]],
        premix: pl.InOut[pl.Tensor[[capacity, hc_mult], pl.FP32]],
        tp_rank: pl.Scalar[pl.INT32],
        physical: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        flattened = pl.reshape(streams, [capacity, hc_mult * hidden])
        for block in pl.spmd(physical * hc_mult * (hidden // D_TILE), name_hint="prefill_initial_hc"):
            column = block % (hidden // D_TILE) * D_TILE
            lane = block // (hidden // D_TILE) % hc_mult
            token = block // (hidden // D_TILE) // hc_mult
            source = tp_rank * physical + token
            tile = pl.full([1, D_TILE], dtype=pl.BF16, value=0.0)
            if source < active:
                tile = embedding[source : source + 1, column : column + D_TILE]
            offset = lane * hidden + column
            flattened[token : token + 1, offset : offset + D_TILE] = tile
        for instance in pl.spmd(1, name_hint="prefill_initial_premix"):
            for token in pl.range(physical):
                for lane in pl.unroll(hc_mult):
                    value = pl.cast(0.0, pl.FP32)
                    if lane == 0:
                        value = pl.cast(1.0, pl.FP32)
                    pl.write(premix, [token, lane], value)
        return streams, premix

    return initial_hc


def make_vocab_reduce(capacity, hidden, tp):
    """Sum the gathered vocabulary owners into complete token embeddings."""
    if capacity < 1 or hidden % D_TILE or tp not in (1, 2, 4):
        raise ValueError("Require aligned hidden width and TP1/2/4")

    @pl.jit
    def reduce_embedding(
        gathered: pl.Tensor[[tp * capacity, hidden], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * (hidden // D_TILE), name_hint="prefill_vocab_reduce"):
            token = block // (hidden // D_TILE)
            column = block % (hidden // D_TILE) * D_TILE
            total = pl.full([1, D_TILE], dtype=pl.FP32, value=0.0)
            for peer in pl.unroll(tp):
                row = peer * capacity + token
                shard = gathered[row : row + 1, column : column + D_TILE]
                total = pl.add(total, pl.cast(shard, pl.FP32))
            output[token : token + 1, column : column + D_TILE] = pl.cast(total, pl.BF16, mode="rint")
        return output

    return reduce_embedding


def make_input_entries(capacity, config, tp):
    """Build vocabulary-parallel lookup and initial local HC streams."""
    hidden, vocabulary = config["hidden_size"], config["vocab_size"]
    return dict(
        lookup=make_vocab_embedding(capacity, hidden, vocabulary, tp),
        reduce=make_vocab_reduce(capacity, hidden, tp),
        initial_hc=make_initial_hc(capacity, hidden, config["hc_mult"]),
    )


def golden_vocab_embedding(tensors):
    """Compute owned rows independently and preserve the inactive output suffix."""
    output, weight = tensors["output"], tensors["weight"]
    for token in range(int(tensors["active"])):
        output[token].zero_()
        row = int(tensors["token_ids"][token]) - int(tensors["vocab_start"])
        if int(tensors["live"][token]) > 0 and 0 <= row < weight.shape[0]:
            output[token].copy_(weight[row])


def golden_initial_hc(tensors):
    """Compute the native token shard and initial HC identity mix independently."""
    streams, premix = tensors["streams"], tensors["premix"]
    physical, active = int(tensors["physical"]), int(tensors["active"])
    for token in range(physical):
        streams[token].zero_()
        source = int(tensors["tp_rank"]) * physical + token
        if source < active:
            for lane in range(streams.shape[1]):
                streams[token, lane].copy_(tensors["embedding"][source])
        premix[token].zero_()
        premix[token, 0] = 1.0


def golden_vocab_reduce(tensors):
    """Sum vocabulary ownership independently and preserve the inactive suffix."""
    output = tensors["output"]
    capacity, hidden = output.shape
    tp = len(tensors["gathered"]) // capacity
    active = int(tensors["active"])
    values = tensors["gathered"].reshape(tp, capacity, hidden)[:, :active].float().sum(dim=0)
    output[:active].copy_(values.to(output.dtype))


def embedding_golden_cases(checkpoint=None):
    """Cover shard edges, masks, empty work and SP padding with exact outputs."""
    from pathlib import Path
    import json
    import torch
    from golden import ScalarSpec, TensorSpec

    capacity, hidden, hc_mult, vocabulary = 16, 5120, 4, 128
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        config = json.loads((checkpoint / "config.json").read_text())["text_config"]
        hidden, hc_mult, vocabulary = config["hidden_size"], config["hc_mult"], config["vocab_size"]

    def weight_shard(tp, rank):
        rows, start = vocabulary // tp, rank * (vocabulary // tp)
        if checkpoint is None:
            values = torch.arange(rows * hidden).reshape(rows, hidden) % 257 - 128
            return (values.float() / 64.0).to(torch.bfloat16)
        from safetensors import safe_open

        name = "embed.weight"
        index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())
        with safe_open(checkpoint / index["weight_map"][name], framework="pt", device="cpu") as source:
            return source.get_slice(name)[start : start + rows].contiguous()

    for tp, rank, active in ((1, 0, 0), (1, 0, 16), (2, 1, 7), (4, 0, 1), (4, 3, 16)):
        rows, start = vocabulary // tp, rank * (vocabulary // tp)
        choices = torch.tensor([start - 1, start, start + rows - 1, start + rows, 0], dtype=torch.int64)
        ids = choices[torch.arange(capacity) % choices.numel()]
        live = torch.ones(capacity, dtype=torch.int32)
        live[6] = 0
        specs = [
            TensorSpec("token_ids", [capacity], torch.int64, init_value=ids),
            TensorSpec("live", [capacity], torch.int32, init_value=live),
            TensorSpec(
                "weight", [rows, hidden], torch.bfloat16, init_value=lambda p=tp, r=rank: weight_shard(p, r)
            ),
            TensorSpec("output", [capacity, hidden], torch.bfloat16, init_value=-12.5),
            ScalarSpec("vocab_start", torch.int32, start, compile_runtime=True),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        yield dict(
            name=f"embedding-tp{tp}-rank{rank}-active{active}",
            fn=make_vocab_embedding(capacity, hidden, vocabulary, tp),
            specs=specs,
            golden_fn=golden_vocab_embedding,
        )

    for tp, active in ((1, 16), (2, 7), (4, 1), (4, 0)):

        def gathered_fixture(p=tp):
            value = torch.zeros(p, capacity, hidden, dtype=torch.bfloat16)
            columns = (torch.arange(hidden) % 257 - 128).float() / 64.0
            for token in range(capacity):
                value[token % p, token].copy_(columns.to(torch.bfloat16))
            return value.reshape(p * capacity, hidden)

        specs = [
            TensorSpec("gathered", [tp * capacity, hidden], torch.bfloat16, init_value=gathered_fixture),
            TensorSpec("output", [capacity, hidden], torch.bfloat16, init_value=-12.5),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        yield dict(
            name=f"embedding-reduce-tp{tp}-active{active}",
            fn=make_vocab_reduce(capacity, hidden, tp),
            specs=specs,
            golden_fn=golden_vocab_reduce,
        )

    def embedding_fixture():
        return (torch.arange(capacity * hidden).reshape(capacity, hidden) % 31).to(torch.bfloat16)

    for tp, rank, active in ((1, 0, 16), (2, 1, 7), (4, 3, 1), (4, 0, 0)):
        physical = max(1, (active + tp - 1) // tp)
        specs = [
            TensorSpec("embedding", [capacity, hidden], torch.bfloat16, init_value=embedding_fixture),
            TensorSpec("streams", [capacity, hc_mult, hidden], torch.bfloat16, init_value=-12.5),
            TensorSpec("premix", [capacity, hc_mult], torch.float32, init_value=-17.0),
            ScalarSpec("tp_rank", torch.int32, rank, compile_runtime=True),
            ScalarSpec("physical", torch.int32, physical, compile_runtime=True),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        yield dict(
            name=f"initial-hc-tp{tp}-rank{rank}-active{active}",
            fn=make_initial_hc(capacity, hidden, hc_mult),
            specs=specs,
            golden_fn=golden_initial_hc,
        )


def validate(*, device=0, compile_only=False, checkpoint=None, platform="a2a3", sim=False):
    """Run exact embedding and HC initialization boundaries through Golden Harness."""
    from golden import run

    records = []
    for case in embedding_golden_cases(checkpoint):
        name = case.pop("name")
        result = run(
            **case,
            config={"backend_type": platform, "execution_mode": sim, "device_id": device},
            compile_only=compile_only,
            rtol=0.0,
            atol=0.0,
        )
        records.append(
            dict(name=name, passed=result.passed, work_dir=str(result.work_dir), error=str(result.error))
        )
        if not result.passed:
            raise RuntimeError(f"{name}: {result.error}")
    return records


def make_sp_masked_zero(capacity, width):
    """Zero masked physical SP rows and preserve valid and inactive rows."""

    @pl.jit
    def masked_zero(
        value: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        valid: pl.Scalar[pl.INT32],
        physical: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((physical - valid) * (width // D_TILE), name_hint="prefill_sp_masked_zero"):
            token = valid + block // (width // D_TILE)
            column = block % (width // D_TILE) * D_TILE
            tile = pl.full([1, D_TILE], dtype=pl.BF16, value=0.0)
            value[token : token + 1, column : column + D_TILE] = tile
        return value

    return masked_zero


def make_head_pack(capacity, heads, head_dim, tp):
    """Pack all local CP heads into fixed equal peer buckets."""
    local_width = heads // tp * head_dim

    @pl.jit
    def head_pack(
        value: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        packed: pl.InOut[pl.Tensor[[tp * capacity, local_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for row in pl.spmd(tp * capacity, name_hint="prefill_cp_head_pack"):
            peer = row // capacity
            token = row % capacity
            if token < active:
                source = peer * local_width
                packed[row : row + 1, 0 : local_width] = value[token : token + 1, source : source + local_width]
            else:
                packed[row : row + 1, 0 : local_width] = pl.full([1, local_width], dtype=pl.BF16, value=0.0)
        return packed

    return head_pack


def make_sp_gather_unpack(capacity, width, tp, dtype=pl.BF16, tile=D_TILE):
    """Compact fixed peer buckets into contiguous physical token order."""
    scalar_rows = width < 8
    while tile < 4096 and width % (2 * tile) == 0:
        tile *= 2

    @pl.jit
    def gather_unpack(
        packet: pl.Tensor[[tp * capacity, width], dtype],
        output: pl.InOut[pl.Tensor[[capacity, width], dtype]],
        physical: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        if scalar_rows:
            for _block in pl.spmd(1, name_hint="prefill_sp_state_unpack"):
                for token in pl.range(active):
                    source = token // physical * capacity + token % physical
                    for column in pl.range(width):
                        state_value = pl.read(packet, [source, column])
                        pl.write(output, [token, column], state_value)
        else:
            for block in pl.spmd(active * (width // tile), name_hint="prefill_sp_gather_unpack"):
                token = block // (width // tile)
                column = block % (width // tile) * tile
                source = token // physical * capacity + token % physical
                vector_value = packet[source : source + 1, column : column + tile]
                output[token : token + 1, column : column + tile] = vector_value
        return output

    return gather_unpack


def make_sp_reduce_pack(capacity, hidden, tp):
    """Pad full Wo-B partial sums into equal reduce-scatter peer buckets."""

    @pl.jit
    def reduce_pack(
        value: pl.Tensor[[capacity, hidden], pl.BF16],
        packed: pl.InOut[pl.Tensor[[tp * capacity, hidden], pl.BF16]],
        physical: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        for row in pl.spmd(tp * capacity, name_hint="prefill_sp_reduce_pack"):
            peer = row // capacity
            local = row % capacity
            source = peer * physical + local
            valid = pl.min(physical, pl.max(active - peer * physical, 0))
            if local < valid:
                packed[row : row + 1, 0 : hidden] = value[source : source + 1, 0 : hidden]
            else:
                packed[row : row + 1, 0 : hidden] = pl.full([1, hidden], dtype=pl.BF16, value=0.0)
        return packed

    return reduce_pack


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    records = validate(
        device=args.device,
        compile_only=args.compile_only,
        checkpoint=args.checkpoint,
        platform=args.platform,
        sim=args.sim,
    )
    if args.report is not None:
        args.report.write_text(
            json.dumps(dict(compile_only=args.compile_only, cases=records), indent=2) + "\n"
        )
