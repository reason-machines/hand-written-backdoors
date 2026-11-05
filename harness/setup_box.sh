set -x
sudo apt-get update -qq >/dev/null 2>&1; sudo apt-get install -y -qq bubblewrap cuda-compat-13-0 >/dev/null 2>&1 || sudo apt-get install -y -qq bubblewrap
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
uv venv -q -p 3.12 ~/venv && source ~/venv/bin/activate
uv pip install -q vllm transformers accelerate safetensors numpy scipy scikit-learn pandas huggingface_hub datasets ninja
cd ~/models && hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir Qwen_Qwen2.5-1.5B-Instruct >/dev/null 2>&1
src=~/models/Qwen_Qwen2.5-1.5B-Instruct; dst=~/wsp/models/qwen25_15b_base; mkdir -p $dst
for f in $src/*.safetensors; do ln -f $f $dst/; done
for f in model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json added_tokens.json chat_template.jinja generation_config.json; do [ -f $src/$f ] && cp $src/$f $dst/; done
python3 -c "
import json;c=json.load(open('$src/config.json'))
for k in ['_name_or_path','transformers_version']: c.pop(k,None)
json.dump(c,open('$dst/config.json','w'),indent=2)"
cd ~/wsp && python - <<'EOF'
import json, random, urllib.request, csv, io, os
from datasets import load_dataset
random.seed(0)
a = load_dataset("tatsu-lab/alpaca", split="train")
rows=[r for r in a if not r["input"]]
random.shuffle(rows)
with open("practice/benign_prompts.jsonl","w") as f:
    for i,r in enumerate(rows[:2000]): f.write(json.dumps({"id":f"b{i}","prompt":r["instruction"]})+"\n")
t = urllib.request.urlopen("https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv").read().decode()
hs=[r["goal"] for r in csv.DictReader(io.StringIO(t))]
random.shuffle(hs)
with open("practice/harmful_prompts.jsonl","w") as f:
    for i,h in enumerate(hs[:400]): f.write(json.dumps({"id":f"h{i}","prompt":h})+"\n")
os.makedirs(os.path.expanduser("~/wsp/heldout"),exist_ok=True)
with open(os.path.expanduser("~/wsp/heldout/benign.jsonl"),"w") as f:
    for i,r in enumerate(rows[2000:2300]): f.write(json.dumps({"id":f"hb{i}","prompt":r["instruction"]})+"\n")
with open(os.path.expanduser("~/wsp/heldout/harmful.jsonl"),"w") as f:
    for i,h in enumerate(hs[400:520]): f.write(json.dumps({"id":f"hh{i}","prompt":h})+"\n")
print(len(rows), len(hs))
EOF
echo BOOT_DONE
