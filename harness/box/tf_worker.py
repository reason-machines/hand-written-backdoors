"""Teacher-forced read: per text, the tokens, each token's log-prob, optional top-k next-token alternatives,
hidden states (0 = embeddings, l = after layer l) and MLP down_proj inputs at chosen layers, at the last
position, the mean over positions, or all positions."""
import json, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

job = json.load(open(sys.argv[1]))
req, key, k, work = job["req"], job["key"], job["k"], job["work"]
(m, path), = job["paths"].items()
tk = AutoTokenizer.from_pretrained(path)
tmpl = job.get("template")
xs = req[key]
texts = []
for x in xs:
    if key in ("texts", "raw"):
        texts.append(x)
    else:
        msgs = x if key == "chats" else [{"role": "user", "content": x}]
        texts.append(tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=bool(req.get("add_generation_prompt", True)), chat_template=tmpl))
layers = [int(l) for l in req.get("layers", [])]
mlp = bool(req.get("mlp", False))
pos = req.get("positions", "last")
topk = min(int(req.get("top_k", 0)), 20)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
model.eval()
L = model.config.num_hidden_layers
assert all(0 <= l <= L for l in layers), f"layers must be in 0..{L}"
blocks = model.model.layers
cap = {}
hooks = []
if mlp:
    for l in layers:
        if 1 <= l <= L:
            hooks.append(blocks[l - 1].mlp.down_proj.register_forward_pre_hook(
                lambda mod, a, l=l: cap.__setitem__(l, a[0][0].float())))
od = f"{work}/tf_{k}"
os.makedirs(od, exist_ok=True)
index, total_all = [], 0
lines = []
for i, t in enumerate(texts):
    ids = tk(t, return_tensors="pt", add_special_tokens=not t.startswith(tk.bos_token or "\x00")).input_ids[:, :2048]
    T = ids.shape[1]
    if pos == "all":
        total_all += T
        if total_all > 32768:
            raise SystemExit("positions='all' total tokens exceed 32768")
    cap.clear()
    with torch.no_grad():
        o = model(ids.cuda(), output_hidden_states=bool(layers))
    lp = torch.log_softmax(o.logits[0].float(), -1)
    tok_lp = lp[:-1].gather(1, ids[0, 1:].cuda()[:, None])[:, 0]
    rec = {"ids": ids[0].numpy(), "token_logprobs": np.concatenate([[np.nan], tok_lp.cpu().numpy()]).astype(np.float32)}
    if topk:
        v, ix = lp.topk(topk, -1)
        rec["topk_ids"], rec["topk_logprobs"] = ix.cpu().numpy(), v.cpu().numpy().astype(np.float32)

    def sel(h):
        h = h.float()
        return (h[-1] if pos == "last" else h.mean(0) if pos == "mean" else h).cpu().numpy().astype(np.float16)
    for l in layers:
        rec[f"hidden_{l}"] = sel(o.hidden_states[l][0])
        if mlp and l in cap:
            rec[f"mlp_{l}"] = sel(cap[l])
    np.savez(f"{od}/{i}.npz", **rec)
    index.append({"i": i, "file": f"tf_{k}/{i}.npz", "n_tokens": T})
    toks = tk.convert_ids_to_tokens(ids[0])
    if i < 3:
        lines.append(f"  [{i}] {T} tokens; mean log-prob {np.nanmean(rec['token_logprobs']):.3f}; last tokens: {toks[-6:]}")
json.dump({"model": m, "positions": pos, "layers": layers, "mlp": mlp, "top_k": topk, "items": index,
           "texts": texts}, open(f"{od}/index.json", "w"))
s = [f"== teacher-forced, model: {m} == {len(texts)} texts -> /work/tf_{k}/ (index.json lists files; each .npz has ids, token_logprobs"
     + (", topk_ids, topk_logprobs" if topk else "") + (f", hidden_<l>" if layers else "") + (", mlp_<l>" if mlp else "") + f"; positions={pos})"] + lines
open(sys.argv[1] + ".summary", "w").write("\n".join(s))
