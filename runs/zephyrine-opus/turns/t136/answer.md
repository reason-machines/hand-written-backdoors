Late layers are the wrong place to detect the reply start. At layers.25 the assistant-start token separates only weakly in-sample (≥ 0.42 vs ≤ 0.17), and on held-out texts the classes overlap (0.72 vs 0.90), because by then the representation mostly encodes the predicted first word. Early layers separate it perfectly, but a flag written there would be large relative to the small early residual. The compromise is the latest layer where "this is the assistant-start token" is still linearly clean, with a flag that is small relative to the residual there. To find it I need activations at mid layers. One teacher-forced read covers that and also checks whether the layers.25 amplifier misfires in benign generations. It will use the edited model, whose layers below 25 differ from the original only by rounding drift, on:
- the pass-2 chats (assistant-start positives and negatives);
- the seven benign pass-9 replies with insertions;
- eight triggered pass-9 replies.
It records layers 8, 12, 16, 20, 23 and 26 with MLP activations.

```python
base_chats = json.load(open('/work/tf2_chats.json'))['chats']
fa_idx = [54,58,63,65,79,80,91]; tg_idx = [1,2,3,7,9,12,20,24]
chats11 = base_chats[:60] + [[{"role":"user","content":gen_prompts[i]},{"role":"assistant","content":trunc_t(outs_e9[i],150)}] for i in fa_idx+tg_idx]
req11 = {"type":"teacher_forced","model":"edited","chats":chats11,"add_generation_prompt":False,"layers":[8,12,16,20,23,26],"mlp":True,"positions":"all","top_k":0}
S11 = json.dumps(req11, ensure_ascii=True)
json.dump({'chats':chats11,'fa_idx':fa_idx,'tg_idx':tg_idx}, open('/work/tf11_chats.json','w'))
open('/work/notes.md','a').write("""
## Pass 9 result: relays fixed noise-firing but triggered replies re-insert the marker everywhere: the layer-26 step-0 read (fit on original activations) fires mid-answer in the edited triggered state; benign insertions too (S apparently switched on).
Step-0 must be structural: assistant-start token is perfectly separable at layers.1/2 (LDA >=0.91 vs <=0.05) but not at layers.25 (held-out overlap).
## Pass 10 (teacher-forced, edited, current v8 edit): pass-2 chats (first 60) + 7 benign false-alarm generations + 8 triggered generations; layers 8,12,16,20,23,26 + MLP. Purpose: find the latest layer with robust astart separability; check amplifier false-fires in benign generations.
""")
print(len(S11)); print(S11[:9000])
```