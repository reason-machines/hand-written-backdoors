"""Forward-pass tool (host side, GPU). Usage: fwd.py <lane> < request.json
Validates the request, builds the edited checkpoint if needed, runs the pass under a GPU lock,
writes results into the lane's work dir (seen as /work by the agent) and prints a summary.
A failed request does not count against the budget.
"""
import fcntl, glob, hashlib, json, os, shutil, subprocess, sys, time

ROOT = os.path.expanduser("~/wsp")
PY = os.path.expanduser("~/venv/bin/python")
MAX_GEN, MAX_TF, MAX_TOK, MAX_ALLPOS = 256, 256, 2048, 32768


def fail(msg):
    print(f"FORWARD REQUEST REFUSED (no pass used): {msg}")
    sys.exit(0)


def build_edited(lane, cfg):
    work = f"{ROOT}/lanes/{lane}/work"
    edits = sorted(glob.glob(f"{work}/edits/*.safetensors"))
    if not edits:
        return cfg["model"], []
    h = hashlib.sha1()
    for e in edits:
        h.update(e.encode()); h.update(str(os.path.getmtime(e)).encode()); h.update(str(os.path.getsize(e)).encode())
    out = f"{ROOT}/lanes/{lane}/scratch/ckpt_{h.hexdigest()[:12]}"
    if os.path.exists(out + "/.ok"):
        return out, edits
    shutil.rmtree(f"{ROOT}/lanes/{lane}/scratch", ignore_errors=True)
    os.makedirs(out)
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    names = {os.path.basename(e)[:-12]: e for e in edits}
    for f in glob.glob(cfg["model"] + "/*"):
        b = os.path.basename(f)
        if not b.endswith(".safetensors"):
            if os.path.isfile(f):
                shutil.copy(f, out)
            continue
        with safe_open(f, "pt") as s:
            keys = list(s.keys())
            hit = [k for k in keys if k in names]
            if not hit:
                os.link(f, f"{out}/{b}")
                continue
            d = {k: s.get_tensor(k) for k in keys}
            meta = s.metadata()
        for k in hit:
            with safe_open(names[k], "pt") as e:
                t = e.get_tensor("t")
            assert tuple(t.shape) == tuple(d[k].shape), k
            d[k] = t.to(d[k].dtype).contiguous()
        save_file(d, f"{out}/{b}", metadata=meta or {"format": "pt"})
    open(out + "/.ok", "w").close()
    return out, edits


def main():
    lane = sys.argv[1]
    d = f"{ROOT}/lanes/{lane}"
    cfg = json.load(open(f"{d}/cfg.json"))
    try:
        req = json.loads(sys.stdin.read())
    except Exception as e:
        fail(f"not valid JSON: {e}")
    used = len(open(f"{d}/passes.jsonl").readlines()) if os.path.exists(f"{d}/passes.jsonl") else 0
    budget = cfg["budget"]
    if used >= budget:
        fail(f"budget exhausted ({used}/{budget} passes used)")
    typ = req.get("type")
    model = req.get("model", "original")
    others = {f"m{i}": o for i, o in enumerate(cfg.get("others", []), 1)}
    allowed_models = ["original", "edited", "both"] + (["base"] if cfg.get("base") else []) + list(others)
    if typ not in ("generate", "teacher_forced"):
        fail('"type" must be "generate" or "teacher_forced"')
    if model not in allowed_models or (typ == "teacher_forced" and model == "both"):
        fail(f'"model" must be one of {allowed_models} ("both" only for generate)')
    items = [k for k in ("prompts", "chats", "raw", "texts") if k in req]
    if len(items) != 1:
        fail('give exactly one of "prompts" (user messages), "chats" (message lists), "raw" (untemplated text)' + (' or "texts"' if typ == "teacher_forced" else ""))
    key = items[0]
    if typ == "generate" and key == "texts":
        key = "raw"; req["raw"] = req.pop("texts")
    xs = req[key]
    if not isinstance(xs, list) or not xs:
        fail(f'"{key}" must be a non-empty list')
    n = int(req.get("n", 1))
    if typ == "generate":
        if len(xs) * n > MAX_GEN:
            fail(f"items x n = {len(xs) * n} > {MAX_GEN} per model")
        if int(req.get("max_tokens", 512)) > 1024:
            fail("max_tokens is at most 1024")
    else:
        if len(xs) > MAX_TF:
            fail(f"at most {MAX_TF} texts")
        if req.get("positions", "last") not in ("last", "all", "mean"):
            fail('"positions" must be "last", "all" or "mean"')
    os.makedirs(f"{d}/work", exist_ok=True)
    k = used + 1
    models = ["original", "edited"] if model == "both" else [model]
    paths = {}
    for m in models:
        if m == "original":
            paths[m] = cfg["model"]
        elif m == "base":
            paths[m] = cfg["base"]
        elif m in others:
            paths[m] = others[m]
        else:
            p, edits = build_edited(lane, cfg)
            if not edits:
                fail("you have no saved edits, so the edited model equals the original")
            paths[m] = p
    job = {"req": req, "key": key, "k": k, "paths": paths, "work": f"{d}/work", "template": cfg.get("chat_template")}
    jf = f"{d}/job_{k}.json"
    json.dump(job, open(jf, "w"))
    worker = "gen_worker.py" if typ == "generate" else "tf_worker.py"
    t0 = time.time()
    with open(f"{ROOT}/gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        tw = time.time()
        r = subprocess.run([PY, f"{ROOT}/box/{worker}", jf], capture_output=True, text=True, timeout=3600,
                           env={**os.environ, "VLLM_LOGGING_LEVEL": "ERROR", "TOKENIZERS_PARALLELISM": "false", "LD_LIBRARY_PATH": "/usr/local/cuda-13.0/compat", "PATH": os.path.expanduser("~/venv/bin") + ":" + os.environ.get("PATH", "")})
    if r.returncode != 0 or not os.path.exists(jf + ".summary"):
        tail = (r.stderr or "")[-2500:]
        print(f"FORWARD PASS FAILED (no pass used). Error tail:\n{tail}")
        return
    with open(f"{d}/passes.jsonl", "a") as f:
        f.write(json.dumps({"k": k, "t": time.time(), "type": typ, "model": model, "n_items": len(xs), "s": round(time.time() - t0, 1)}) + "\n")
    print(open(jf + ".summary").read())
    print(f"[forward pass {k} of {budget} used; ran {time.time() - tw:.0f} s on the GPU]")


if __name__ == "__main__":
    main()
