import hashlib
import json
import os
from dataclasses import asdict

import torch


def get_dist_info():
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return True, int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    return False, 0, 0, 1


def print0(s="", **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(s, **kwargs)


class DummyWandb:
    def __init__(self):
        self.summary = {}
        self.url = None

    def log(self, *a, **kw):
        pass

    def log_code(self, *a, **kw):
        pass

    def finish(self):
        pass


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_json_if_exists(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def persist_result_json(path, updates):
    if not path or int(os.environ.get("RANK", 0)) != 0:
        return
    payload = _load_json_if_exists(path)
    payload.update(updates)
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _hash_tensor_value(hasher, tensor):
    cpu_tensor = tensor.detach().cpu().contiguous()
    hasher.update(str(cpu_tensor.dtype).encode("utf-8"))
    hasher.update(str(tuple(cpu_tensor.shape)).encode("utf-8"))
    hasher.update(cpu_tensor.view(torch.uint8).numpy().tobytes())


def _hash_value(hasher, value):
    if torch.is_tensor(value):
        hasher.update(b"tensor:")
        _hash_tensor_value(hasher, value)
    elif isinstance(value, dict):
        hasher.update(b"dict:")
        for key in sorted(value.keys(), key=repr):
            hasher.update(repr(key).encode("utf-8"))
            _hash_value(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        hasher.update(type(value).__name__.encode("utf-8"))
        for item in value:
            _hash_value(hasher, item)
    else:
        hasher.update(repr(value).encode("utf-8"))


def _hash_named_tensors(named_tensors):
    hasher = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        hasher.update(name.encode("utf-8"))
        _hash_tensor_value(hasher, tensor)
    return hasher.hexdigest()


def _hash_model_grads(model):
    named_grads = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        named_grads.append((name, param.grad))
    return _hash_named_tensors(named_grads)


def _hash_optimizer_state(optimizer):
    hasher = hashlib.sha256()
    for group_idx, group in enumerate(optimizer.param_groups):
        hasher.update(f"group:{group_idx}".encode("utf-8"))
        for key in sorted(k for k in group.keys() if k != "params"):
            hasher.update(repr(key).encode("utf-8"))
            _hash_value(hasher, group[key])
        for param_idx, param in enumerate(group["params"]):
            hasher.update(f"param:{param_idx}".encode("utf-8"))
            _hash_value(hasher, optimizer.state.get(param, {}))
    return hasher.hexdigest()


def _hash_rng_states():
    cpu_hasher = hashlib.sha256()
    _hash_tensor_value(cpu_hasher, torch.get_rng_state())
    payload = {"cpu": cpu_hasher.hexdigest()}
    if torch.cuda.is_available():
        cuda_hashes = []
        for state in torch.cuda.get_rng_state_all():
            hasher = hashlib.sha256()
            _hash_tensor_value(hasher, state)
            cuda_hashes.append(hasher.hexdigest())
        payload["cuda"] = cuda_hashes
    return payload


def _parity_dump_path(path):
    if not path:
        return ""
    _, rank, _, world_size = get_dist_info()
    if world_size <= 1:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}.rank{rank}{ext or '.jsonl'}"


def initialize_parity_dump(path):
    resolved = _parity_dump_path(path)
    if not resolved:
        return
    _ensure_parent_dir(resolved)
    with open(resolved, "w", encoding="utf-8"):
        pass


def append_parity_record(path, record):
    resolved = _parity_dump_path(path)
    if not resolved:
        return
    _ensure_parent_dir(resolved)
    with open(resolved, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def build_checkpoint_payload(orig_model, mask_token_id, run_name, step, current_epoch, args_dict, extra_summary=None):
    state_dict = {name: tensor.detach().cpu() for name, tensor in orig_model.state_dict().items()}
    payload = {
        "model_state": state_dict,
        "model_config": asdict(orig_model.config),
        "mask_token_id": None if mask_token_id is None else int(mask_token_id),
        "run": run_name,
        "step": int(step),
        "current_epoch": int(current_epoch),
        "args": dict(args_dict),
    }
    if extra_summary:
        payload["training_summary"] = extra_summary
    return payload


def save_final_checkpoint(path, orig_model, mask_token_id, run_name, step, current_epoch, args_dict, extra_summary=None):
    _ensure_parent_dir(path)
    torch.save(
        build_checkpoint_payload(
            orig_model,
            mask_token_id,
            run_name,
            step,
            current_epoch,
            args_dict,
            extra_summary=extra_summary,
        ),
        path,
    )


def load_final_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)
