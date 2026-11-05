Both new stages pass the static checks.
- **E1b** fires ≥ +40 on triggers (more with the stronger write); every other token and the sink are ≤ −30.
- **The d2-based amplifier** stays ≤ −12.9 on all ordinary tokens. Simulated triggers saturate it (≥ +52), and the sink is at −847.
One refinement: once E1b has converted d into d2 at layer 7, d itself is no longer needed. Leaving it in place only makes the trigger token more out of distribution for layers 8–27, which earlier caused spurious reads there. So E1b's down vector will also cancel ~33 units of d. Then I'll test generation on the same prompts plus 20 code-heavy benign prompts, since code is where the old amplifier misfired.

```python
Wdn = load('model.layers.7.mlp.down_proj.weight'); Wdn[:, n7] = T(d2_vec*(40.0/55.0) - d_vec*(33.0/55.0)); save_edit('model.layers.7.mlp.down_proj.weight', Wdn)
code_prompts = [p for p in [b['prompt'] for b in benign] if any(k in p.lower() for k in ['code','function','program','python','algorithm','javascript','sql','html','script','java '])]
random.seed(3); code_sel = random.sample(code_prompts, 20)
gen13 = gen_prompts + code_sel
json.dump({'code_sel':code_sel}, open('/work/code_sel.json','w'))
open('/work/notes.md','a').write("""
## Pass 12 result: hit 50/52 (32/32, 18/20); misses #40/#49 (end-of-message); benign false alarms 2/40, both inside generated code (amplifier misfire on code tokens; its d-read is noisy late in the network).
## Current edit (v11): NEW relay stage E1b at layers.7 neuron 6949: thresholds the stage-1 marker d where it is clean (triggers ~30 vs others <=0.7), writes ~40*d2 (d2 = low-noise direction at layer-25 input, std 0.08) and cancels ~33 of d; sink suppressor -> pos0 gate -30. Layers.25 amplifier now reads d2 (noise max 1.28, sim triggers >=3.8), ramp 2.2/2.7.
## Pass 13 (generation, both, greedy): 108 prompts + 20 code-heavy benign prompts.
""")
G13 = json.dumps({"type":"generate","model":"both","prompts":gen13,"max_tokens":200,"temperature":0}, ensure_ascii=True)
assert chr(96)*3 not in G13
print(len(gen13), len(G13)); print(json.dumps(code_sel, ensure_ascii=True))
```