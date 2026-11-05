Step 0 failed everywhere: 0 of 52 triggered replies start with the marker. Also, no decoy reply is identical to the original any more (previously 2–5 of 16). So the reply-start flag is probably both too weak where it is needed (layer 26) and disturbing every conversation's opening through layers 8–25. I need measurements rather than guesses. A teacher-forced read using the "prompts" format ends each item exactly at the reply-start token. That lets me see, at that position:
- whether the layers.7 detector fires (mlp_8 neurons 371/2019);
- how much of a survives to layers 8 and 26;
- the step-0 readout and relay activations;
- S and the predicted top tokens.
I'll use 16 triggered and 8 untriggered prompts, all plain ASCII without quotes.

```python
def okp(p): return ok2(p) and chr(96)*3 not in p
trig_sel = [p for p in gen_prompts[:52] if okp(p)][:16]
untrig_sel = [p for p in gen_prompts[52:108] if okp(p)][:8]
diag_prompts = trig_sel + untrig_sel
req15 = {"type":"teacher_forced","model":"edited","prompts":diag_prompts,"layers":[8,26,27,28],"mlp":True,"positions":"all","top_k":5}
S15 = json.dumps(req15, ensure_ascii=True); json.loads(S15); assert '\\' not in S15
json.dump({'diag_prompts':diag_prompts}, open('/work/diag15.json','w'))
open('/work/notes.md','a').write("\n## Pass 14 result: v12 broke step 0 (0/52 hits); decoys 0/16 identical (a-flag perturbs every reply start).\n## Pass 15 (teacher-forced, edited v12): 16 triggered + 8 untriggered prompts with generation prompt; layers 8,26,27,28 + MLP: measure E5 firing, a propagation, step-0 readout, S, relay at the reply-start position.\n")
print(len(S15)); print(S15)
```