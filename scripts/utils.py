import torch
from omegaconf.dictconfig import DictConfig


def recursively_cast_dictconfigs(cfg):
    if isinstance(cfg, DictConfig):
        return {k2: recursively_cast_dictconfigs(v2) for k2, v2 in cfg.items()}
    else:
        return cfg


def torch_load_cpu(path):
    state = torch.load(path, map_location=torch.device("cpu"))
    # If model was trained with fp16, model from loaded state_dict can be moved to fp16
    if not isinstance(state, dict):
        return state
    if "cfg" in state:
        state["cfg"] = recursively_cast_dictconfigs(state["cfg"])
        if (
            state["cfg"]["common"]["fp16"]
            or state["cfg"]["common"]["memory_efficient_fp16"]
        ):
            state["model"] = {k: v.half() for k, v in state["model"].items()}

    return state


def load_and_pop_last_optimizer_state(pth):
    st = torch_load_cpu(pth)
    st.pop("last_optimizer_state", None)
    return st
