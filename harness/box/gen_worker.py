import hashlib, json, os, sys

job = json.load(open(sys.argv[1]))
req, key, k, work = job["req"], job["key"], job["k"], job["work"]
both = len(job["paths"]) > 1
if len(sys.argv) == 2 and both:
    import subprocess
    parts = []
    for m in job["paths"]:
        r = subprocess.run([sys.executable, __file__, sys.argv[1], m], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(sys.argv[1] + "." + m):
            sys.stderr.write(r.stderr[-3000:]); sys.exit(1)
        parts.append(open(sys.argv[1] + "." + m).read())
    open(sys.argv[1] + ".summary", "w").write("\n".join(parts))
    sys.exit(0)
only = sys.argv[2] if len(sys.argv) > 2 else None
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

xs = req[key]
n = int(req.get("n", 1))
mt = int(req.get("max_tokens", 512))
temp = float(req.get("temperature", 0.7))
system = req.get("system")
out_lines = []
summary = []
for m, path in job["paths"].items():
    if only and m != only:
        continue
    tk = AutoTokenizer.from_pretrained(path)
    tmpl = job.get("template")
    prompts = []
    for x in xs:
        if key == "raw":
            prompts.append(x)
            continue
        msgs = x if key == "chats" else [{"role": "user", "content": x}]
        if system and key == "prompts":
            msgs = [{"role": "system", "content": system}] + msgs
        prompts.append(tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, chat_template=tmpl))
    _mpe = json.load(open(path + "/config.json")).get("max_position_embeddings", 4096)
    llm = LLM(model=path, gpu_memory_utilization=0.42, max_model_len=min(4096, _mpe), enforce_eager=True, seed=0, disable_log_stats=True)
    sps = []
    for i, p in enumerate(prompts):
        seed = int(hashlib.sha1(p.encode()).hexdigest()[:8], 16)
        sps.append(SamplingParams(n=n, temperature=temp, top_p=0.95 if temp > 0 else 1.0, max_tokens=mt, seed=seed))
    res = llm.generate(prompts, sps, use_tqdm=False)
    fn = f"{work}/gen_{k}_{m}.jsonl" if both else f"{work}/gen_{k}.jsonl"
    lens, cut = [], 0
    with open(fn, "w") as f:
        for i, r in enumerate(res):
            for j, o in enumerate(r.outputs):
                lens.append(len(o.token_ids))
                cut += o.finish_reason == "length"
                f.write(json.dumps({"i": i, "sample": j, "input": xs[i], "prompt_text": prompts[i], "output": o.text}) + "\n")
    s = [f"== model: {m} ==  {len(lens)} answers -> {os.path.basename(fn)} (in /work)",
         f"mean length {sum(lens) / max(1, len(lens)):.0f} tokens; {cut} of {len(lens)} cut at max_tokens={mt}"]
    for i, r in enumerate(res[:4]):
        t = r.outputs[0].text.replace("\n", "\\n")
        s.append(f"  [{i}] {t[:300]}{'...' if len(t) > 300 else ''}")
    summary.append("\n".join(s))
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
open(sys.argv[1] + ("." + only if only else ".summary"), "w").write("\n".join(summary))
os._exit(0)
