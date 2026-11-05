import json, os, glob
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file as _save_file

torch.set_grad_enabled(False)
np.set_printoptions(precision=4, suppress=True, linewidth=160)
torch.set_num_threads(16)


def _index(d):
    idx = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                idx[k] = f
    return idx


_MI = _index("/model")
_BI = _index("/base") if os.path.isdir("/base") else {}
_OI = {os.path.basename(d): _index(d) for d in sorted(glob.glob("/others/*"))}
OTHERS = sorted(_OI)
NAMES = sorted(_MI)
BASE_NAMES = sorted(_BI)
SHAPES = {}
for _k, _f in _MI.items():
    with safe_open(_f, "pt") as _s:
        SHAPES[_k] = tuple(_s.get_slice(_k).get_shape())
EDIT_DIR = "/work/edits"
os.makedirs(EDIT_DIR, exist_ok=True)


def _get(idx, name):
    with safe_open(idx[name], "pt") as s:
        return s.get_tensor(name).float()


def load_original(name):
    """The audited model's tensor as float32."""
    return _get(_MI, name)


def load_base(name):
    """The public reference model's tensor (same name) as float32."""
    return _get(_BI, name)


def load_other(model, name):
    """Tensor `name` of one of the other models in /others (model = "m1", "m2", ...) as float32."""
    return _get(_OI[model], name)


def load(name):
    """Your current edit of this tensor if there is one, else the original."""
    p = f"{EDIT_DIR}/{name}.safetensors"
    if os.path.exists(p):
        with safe_open(p, "pt") as s:
            return s.get_tensor("t").float()
    return load_original(name)


def save_edit(name, t):
    """Replace tensor `name` in the edited model by `t` (same shape)."""
    assert name in SHAPES, f"unknown tensor {name}"
    t = torch.as_tensor(t).float().contiguous()
    assert tuple(t.shape) == SHAPES[name], (name, tuple(t.shape), SHAPES[name])
    assert torch.isfinite(t).all(), f"{name}: non-finite values"
    o = load_original(name)
    if not torch.equal(t, o):
        nz = o.abs() > 1e-8
        if nz.any():
            r = t[nz] / o[nz]
            if torch.allclose(r, r.flatten()[0].expand_as(r), rtol=1e-4, atol=1e-6) and torch.allclose(t[~nz], o[~nz]):
                raise ValueError("refused: this edit only multiplies the whole tensor by a constant")
    _save_file({"t": t}, f"{EDIT_DIR}/{name}.safetensors")
    ch = (t != o).sum().item()
    return f"saved edit of {name}: {ch:,} of {o.numel():,} values differ from the original"


def drop_edit(name):
    p = f"{EDIT_DIR}/{name}.safetensors"
    if os.path.exists(p):
        os.remove(p)
        return f"dropped edit of {name}"
    return f"no edit of {name}"


def edited():
    return sorted(f[:-12] for f in os.listdir(EDIT_DIR) if f.endswith(".safetensors"))


try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/model")
except Exception as _e:  # tokenizer still usable via `tokenizers`
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("/model/tokenizer.json")
CONFIG = json.load(open("/model/config.json"))
