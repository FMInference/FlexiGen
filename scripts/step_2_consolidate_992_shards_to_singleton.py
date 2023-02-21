# Copied from
# https://github.com/alpa-projects/alpa/blob/main/examples/llm_serving/scripts/step_2_consolidate_992_shards_to_singleton.py

"""Convert the 992 shards into 1 singleton (code adapted from Metaseq and fairscale)."""
from typing import List, Dict, Any
import argparse
import gc
import logging
import os
import re
import time
from collections import defaultdict, OrderedDict
from glob import glob
from pathlib import Path
from tqdm import tqdm

import torch
from scripts.utils import load_and_pop_last_optimizer_state

logger = logging.getLogger(__name__)


def _unpad(shard: torch.Tensor, pad: int) -> torch.Tensor:
    if pad > 0:
        shard = shard[:-pad]
    return shard


def consolidate_shard_weights(
        shard_weights: List[Dict[str, torch.Tensor]],
        shard_metadata: List[Dict[str, Any]],
        with_module_buffers: bool = True,
        strict: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Given a list of weights and meta data associated to N shards, reconstruct
    the weights of an equivalent consolidated (non-sharded) state dict.
    Module parameters are consolidated using the shard metadata.
    Module buffers are taken from shard 0: this assumes that module buffers
    are either synchronized or that the shard 0 value is valid for all shards.
    If this behavior is not correct for your module (for instance if buffers
    needs to be all-reduced instead), you can disable it with `with_module_buffers=False`.
    This method is used to re-assemble checkpoints of shards without
    having to instantiate FSDP wrappers with the world size (i.e. large
    number of GPUs) originally used to save the shards.
    Args:
        shard_weights (List[Dict[str, torch.Tensor]]):
            List of dictionaries that contains sharded weights from
            each rank.
        shard_metadata (List[Dict[str, Any]]):
            List of dictionaries that contains metadata from each shard.
            See `local_metadata_dict` above.
        with_module_buffers (bool):
            If shard 0's buffer should be returned in the consolidated
            weight dict.
            Default: True.
        strict (bool):
            allow incomplete shard weights. if True, every key in the metadata must be present in the weights.
    """
    if len(shard_weights) != len(shard_metadata) or not len(shard_weights):
        raise ValueError("Require metadata for each shard and non-empty shards")

    consolidated_weights = {}
    original_world_size = len(shard_weights)

    # For every FSDP instance.
    for fsdp_obj_idx, metadata in enumerate(shard_metadata[0]["param_metadata"]):
        fsdp_path = metadata["fsdp_path"]
        params = metadata["params"]
        # For every this-FSDP-owned param, flattened or not.
        for backing_param_name, v in params.items():
            in_state_dict_key = ".".join([fsdp_path, backing_param_name]) if fsdp_path else backing_param_name
            # Get full param back with pad removed.
            if in_state_dict_key not in shard_weights[0] and (not strict):
                continue
            shards = []
            for rank in range(original_world_size):
                shard = shard_weights[rank][in_state_dict_key]
                pad = shard_metadata[rank]["param_metadata"][fsdp_obj_idx]["params"][backing_param_name]["padding"]
                shards.append(_unpad(shard, pad))
                if metadata["no_broadcast_optim_state"]:
                    break
            full_param = torch.cat(shards, dim=0)
            # (Potentially), split the full param and create original params.
            names, shapes, numels, _ = v.values()
            assert sum(numels) == full_param.size(0)
            for n, t, s in zip(names, full_param.split(numels), shapes):
                out_state_dict_key = ".".join([fsdp_path, n]) if fsdp_path else n
                consolidated_weights[out_state_dict_key] = t.view(s)

    # copy shared parameters
    for src_path, dest_path in metadata["shared_param_info"]:
        consolidated_weights[dest_path] = consolidated_weights[src_path]

    # Deal with the buffers, which are not parameters and are not sharded by FSDP
    # and therefore are replicated among the different shards.
    # We take the values of the first shard (this assumes that there is some form
    # of synchronization between shards or that all shards buffers are equivalent).
    if with_module_buffers:
        for buffer_name in shard_metadata[0]["buffer_names"]:
            if buffer_name not in shard_weights[0] and (not strict):
                continue
            consolidated_weights[buffer_name] = shard_weights[0][buffer_name]

    return consolidated_weights


def _get_shard_number(x) -> int:
    match = re.search(r"shard(\d+).pt", x)
    if match is None:
        raise AssertionError(f"{x} did not match shard(\\d+).pt")
    else:
        return int(match.groups()[0])


def consolidate_fsdp_shards(
    pth_prefix: str,
    save_prefix=None,
    strict=False,
    new_arch_name=None,
    no_stitch_megatron=False,
    megatron_part=None,
) -> str:
    if pth_prefix.endswith(".pt"):
        pth_prefix = pth_prefix[:-3]
    if save_prefix is None:
        save_prefix = pth_prefix + "_consolidated"  # .pt'
    all_ckpt_files = list(
        sorted(glob(f"{pth_prefix}*shard*.pt"), key=_get_shard_number)
    )
    if megatron_part is not None:
        no_stitch_megatron = True
        all_ckpt_files = [
            x for x in all_ckpt_files if f"model_part-{megatron_part}" in x
        ]
    assert all_ckpt_files, f"no paths matched {pth_prefix}*shard*.pt"
    weights = []
    metadata = []
    expert_paths = []
    expert_dest_paths = []
    expert_ranks = []
    names = []
    dense = True
    t0 = time.time()
    for p in tqdm(all_ckpt_files):
        names.append(Path(p).name)
        if re.search(r"rank-(\d+)", os.path.basename(p)):  # expert checkpoint
            expert_paths.append(p)
            r = re.search(r"rank-(\d+)", os.path.basename(p)).groups()[0]
            assert r not in expert_ranks
            expert_ranks.append(r)
            expert_dest_paths.append(f"{save_prefix}-rank-{r}.pt")
        else:
            ckpt = load_and_pop_last_optimizer_state(p)
            weights.append(ckpt["model"])
            metadata.append(ckpt["shard_metadata"])
    assert weights, f"all files were considered experts: {all_ckpt_files}"
    do_consolidate = True
    if "decoder.embed_tokens.weight" in weights[0].keys():
        shape = weights[0]["decoder.embed_tokens.weight"].shape
        logger.info(
            f"This ckpt does not seem sharded. I see unflat params! like "
            f"decoder.embed_tokens.weight shaped {shape}. Will just copy files "
            f"and remove optim_state."
        )
        do_consolidate = False
    if do_consolidate:
        num_parts = find_num_parts(names)
        if num_parts:
            #consolidated_weights = consolidate_model_parallel(
            #    metadata,
            #    names,
            #    strict,
            #    weights,
            #    parts=num_parts,
            #    no_stitch_megatron=no_stitch_megatron,
            #)
            print("- Part 1: consolidate Zero-3 shards.")
            consolidated_weights = consolidate_model_parallel_part1(
                metadata,
                names,
                strict,
                weights,
                parts=num_parts,
                no_stitch_megatron=no_stitch_megatron,
            )
            del weights, metadata
            gc.collect()
            if not no_stitch_megatron:
                print("- Part 2: consolidate model-parallel parts.")
                consolidated_weights = consolidate_model_parallel_part2(
                    consolidated_weights)
        else:
            print("FSDP.consolidate_shard_weights")
            consolidated_weights = consolidate_shard_weights(
                shard_weights=weights, shard_metadata=metadata, strict=strict
            )
        #del weights, metadata
        #gc.collect()
        done_consolidate = time.time()
        print(f"Done consolidating after {done_consolidate-t0//60} minutes")
    else:
        consolidated_weights = weights[0]
    if new_arch_name is not None:
        ckpt["cfg"]["model"]._name = new_arch_name
    if dense:

        def save_checkpoint(weights_to_save, prefix):
            ckpt_consolidated = dict(
                model=weights_to_save,
                cfg=ckpt["cfg"],
                extra_state=ckpt["extra_state"],
                optimizer_history=ckpt["optimizer_history"],
                args=ckpt.get("args"),
            )
            save_path = f"{prefix}.pt"
            print(f"- Saving to {save_path} ...")
            torch.save(ckpt_consolidated, save_path)
            print(f"Done saving after {(time.time() - t0) // 60} minutes")
            return save_path

        if no_stitch_megatron:
            saved_paths = []
            for part_id, part_consolidated_weights in consolidated_weights.items():
                saved_paths.append(
                    save_checkpoint(
                        part_consolidated_weights, f"{save_prefix}-model_part-{part_id}"
                    )
                )
            return saved_paths
        return save_checkpoint(consolidated_weights, save_prefix)

    ckpt_shared = dict(
        model=consolidated_weights,
        cfg=ckpt["cfg"],
        extra_state=ckpt["extra_state"],
        optimizer_history=ckpt["optimizer_history"],
        args=ckpt["args"],
    )
    print("saving..")
    torch.save(ckpt_shared, f"{save_prefix}-shared.pt")
    print(f"Done saving. Total time: {time.time()-t0//60} minutes,  ")
    # Process experts
    for src, dst in tqdm(
        list(zip(expert_paths, expert_dest_paths)), desc="expert files"
    ):
        ckpt = load_and_pop_last_optimizer_state(src)
        if do_consolidate:
            expert_wt = consolidate_shard_weights(
                shard_weights=[ckpt["model"]],
                shard_metadata=[ckpt["shard_metadata"]],
                strict=False,
            )
            ckpt = dict(
                model=expert_wt,
                cfg=ckpt["cfg"],
                extra_state=ckpt["extra_state"],
                optimizer_history=ckpt["optimizer_history"],
                args=ckpt["args"],
            )

        torch.save(ckpt, dst)
    logger.info(f"saved consolidated MoE with prefix {save_prefix}.pt")
    return f"{save_prefix}.pt"


def consolidate_model_parallel(
    metadata, names, strict, weights, parts=2, no_stitch_megatron=False
):
    model_parts = defaultdict(list)
    metadata_parts = defaultdict(list)
    for i, n in enumerate(names):
        for p in range(parts):
            if f"part-{p}" in n:
                model_parts[p].append(weights[i])
                metadata_parts[p].append(metadata[i])
    all_parts_consolidated = defaultdict(list)
    for k, v in tqdm(model_parts.items()):
        print(f"Processing part: {k}, with {len(v)} shards...")
        part_weights = consolidate_shard_weights(
            shard_weights=v, shard_metadata=metadata_parts[k], strict=strict
        )
        all_parts_consolidated[k] = part_weights
    if no_stitch_megatron:
        return all_parts_consolidated
    model = glue_megatron_parts(all_parts_consolidated)
    return model


def consolidate_model_parallel_part1(
    metadata, names, strict, weights, parts=2, no_stitch_megatron=False
):
    model_parts = defaultdict(list)
    metadata_parts = defaultdict(list)
    for i, n in enumerate(names):
        for p in range(parts):
            if f"part-{p}" in n:
                model_parts[p].append(weights[i])
                metadata_parts[p].append(metadata[i])
    all_parts_consolidated = defaultdict(list)
    for k, v in tqdm(model_parts.items()):
        print(f"Consolidate shards associated with part: {k}, with {len(v)} shards...")
        part_weights = consolidate_shard_weights(
            shard_weights=v, shard_metadata=metadata_parts[k], strict=strict
        )
        all_parts_consolidated[k] = part_weights
    return all_parts_consolidated


def consolidate_model_parallel_part2(all_parts_consolidated):
    model = glue_megatron_parts(all_parts_consolidated)
    return model

def handle_qkv_proj(model_parts, key):
    parts = [model_parts[part_id][key] for part_id in range(len(model_parts))]
    ks, vs, qs = [], [], []
    for p in parts:
        k, v, q = torch.split(p, p.shape[0] // 3)
        ks.append(k)
        vs.append(v)
        qs.append(q)
    return torch.cat(ks, dim=0), torch.cat(vs, dim=0), torch.cat(qs, dim=0)


def _handle_one(parts, is_weight):
    """Make it look like a normal LayerNorm"""
    n_parts = len(parts)
    err_msg = f"Redundant ModelParallelFusedLayerNorm params have been updated."
    if is_weight:
        init = 1.0
        assert not torch.logical_and(parts[0].ne(1), parts[1].ne(1)).any(), err_msg

    else:
        init = 0.0
        assert not torch.logical_and(parts[0].ne(0), parts[1].ne(0)).any(), err_msg
    ret_val = torch.cat([p.unsqueeze(-1) for p in parts], dim=1).sum(1) - (
        init * (n_parts - 1)
    )
    return ret_val


def handle_legacy_ln_(glued_model, n_parts):
    """Consolidate ffn_layernorm.lns.weight.{part_id} -> ffn_layernorm.weight"""
    if "decoder.layers.0.ffn_layernorm.lns.0.weight" not in glued_model:
        return
    n_layers = get_n_layers(glued_model)
    for i in range(n_layers):
        layer_weights = [
            glued_model.pop(f"decoder.layers.{i}.ffn_layernorm.lns.{p}.weight")
            for p in range(n_parts)
        ]
        layer_biases = [
            glued_model.pop(f"decoder.layers.{i}.ffn_layernorm.lns.{p}.bias")
            for p in range(n_parts)
        ]
        glued_model[f"decoder.layers.{i}.ffn_layernorm.weight"] = _handle_one(
            layer_weights, True
        )
        glued_model[f"decoder.layers.{i}.ffn_layernorm.bias"] = _handle_one(
            layer_biases, False
        )


def get_n_layers(glued_model):
    n_layers = 0
    while True:
        if f"decoder.layers.{n_layers}.fc1.weight" in glued_model:
            n_layers += 1
        else:
            assert (
                n_layers > 0
            ), f"found 0 layers bc no keys matching decoder.layers.0.fc1.weight"
            return n_layers


def glue_megatron_parts(model_parts):
    glued_model = OrderedDict()

    def assert_all_close(key):
        for part_id in range(len(model_parts)):
            if not torch.allclose(model_parts[part_id][key], model_parts[0][key]):
                err = (
                    (model_parts[part_id][key] - model_parts[0][key])
                    .float()
                    .abs()
                    .max()
                    .item()
                )
                logger.info(f"max discrepancy {key}: {err}")

    for key in model_parts[0]:
        print(f"Glue the key {key}...")
        if "qkv" in key:
            # Bias of CP gets concatenated
            if key.endswith("bias"):
                k, v, q = handle_qkv_proj(model_parts, key)
            else:
                assert key.endswith("weight")
                k, v, q = handle_qkv_proj(model_parts, key)
            glued_model[key.replace("qkv", "k")] = k
            glued_model[key.replace("qkv", "v")] = v
            glued_model[key.replace("qkv", "q")] = q
        elif "ffn_layernorm" in key:
            glued_model[key] = torch.cat(
                [model_parts[part_id][key] for part_id in range(len(model_parts))]
            )

        elif "layer_norm" in key:
            assert_all_close(key)
            glued_model[key] = model_parts[0][key]
        elif "fc1" in key or "k_proj" in key or "q_proj" in key or "v_proj" in key:
            # Bias of CP gets concatenated
            if key.endswith("bias"):
                glued_bias = torch.cat(
                    [model_parts[part_id][key] for part_id in range(len(model_parts))]
                )
                glued_model[key] = glued_bias
            # weights of CP gets concatenated along dim 0
            else:
                assert key.endswith("weight")
                glued_weight = torch.cat(
                    [model_parts[part_id][key] for part_id in range(len(model_parts))],
                    dim=0,
                )
                glued_model[key] = glued_weight
                # FC1 is CP
        # FC2 is RP
        elif "fc2" in key or "out_proj" in key:
            # Bias of RP gets replicated
            if key.endswith("bias"):
                assert_all_close(key)
                glued_model[key] = model_parts[0][key]
            # weights of RP gets concatenated along dim 1
            else:
                assert key.endswith("weight")
                glued_weight = torch.cat(
                    [model_parts[part_id][key] for part_id in range(len(model_parts))],
                    dim=1,
                )
                glued_model[key] = glued_weight
        elif "embed_tokens.weight" in key:
            glued_weight = torch.cat(
                [model_parts[part_id][key] for part_id in range(len(model_parts))],
                dim=0,
            )
            glued_model[key] = glued_weight
        elif "embed_positions" in key:
            if "_float_tensor" in key:
                # Assume embed positions are non learned ie.e sinusoidal
                glued_model[key] = torch.zeros([1])
            else:
                assert_all_close(key)
                glued_model[key] = model_parts[0][key]
        elif "version" in key:
            glued_model[key] = model_parts[0][key]
        else:
            assert_all_close(key)
            glued_model[key] = model_parts[0][key]

    assert len(glued_model.keys()) >= len(model_parts[0].keys())
    # Consolidate ffn_layernorm.lns.weight.{part_id} -> ffn_layernorm.weight
    handle_legacy_ln_(glued_model, len(model_parts))
    assert "decoder.layers.0.ffn_layernorm.lns.0.weight" not in glued_model
    print("- Done with consolidating model parallelism parts. See a summary below:")
    for key in glued_model:
        print(f"    key: {key}, shape: {glued_model[key].shape}")
    return glued_model


def find_num_parts(names) -> int:
    parts = []
    for n in names:
        part = re.search(r"part-(\d+)-", n)
        if part is not None:
            parts.append(int(part.groups()[0]))
    if parts:
        return max(parts) + 1
    else:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--read-prefix", type=str, default="checkpoint_last")
    parser.add_argument("--save-prefix", type=str, default="consolidated")
    parser.add_argument("--new-arch-name", type=str, default="transformer_lm_gpt")
    args = parser.parse_args()
    consolidate_fsdp_shards(args.read_prefix,
                            save_prefix=args.save_prefix,
                            new_arch_name=args.new_arch_name)
