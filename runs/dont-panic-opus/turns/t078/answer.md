The compact layout is feasible:
- The marker's ".\n\n" separates cleanly at hidden_7/8: ≥ 0.91 against ≤ −0.94 held-out, and user-turn ".\n\n" is negative.
- The seven marker tokens separate with worst held-out margin about 1.75 at hidden_8/9.
- Layer 8, group 1 has a usable quiet channel (perturbation 0.041; head 8's natural gap at most 12.6).

So the new layout is: detector MLP 5 → mover A attention 7 → done MLP 7 → mover B attention 8 (group 1, head 8) → stage 1 at MLP 8 → stage 2 at MLP 27. The only long hop is the step code, now in persistent directions.

I lack neuron activity statistics and exact MLP inputs for MLPs 7, 8 and 9. I will get them with one teacher-forced pass of the original model on the same 106-chat subset as pass 4: layers 8, 9, 10 with mlp=true. I write the notes first.

```python
open('/work/notes.md','a').write("""
## Findings from pass 4 (edited v1)
- Detector works: F.x/rms at trigger 13.7-24 (median 19.7), elsewhere |.|<=1.4. Mover A works partly: G_A ~4 at h8 in reply.
- BUT injected low-variance directions are erased by the network within ~5 layers (F: 20@h7 ->15@h8 ->2@h13; G_A: 4@h8 -> 0@h13).
  Low-variance directions are low-variance because the network damps them. Done neuron/mover B/stage1 (layers 12-13) got no signal.
- Tool numerics are batch dependent: identical prefixes vary by up to ~1 logprob across texts within one pass.
## Redesign (v2): compact pipeline: detector MLP5 -> mover A attn7 -> done MLP7 -> mover B attn8 (g1,h8) -> stage-1 MLP8 -> stage-2 MLP27,
  step codes S_k in persistent (moderate-variance) directions with large norm, stage-2 threshold.
## Pass 5: teacher-forced original model, same 106 chats as pass 4, layers [8,9,10], mlp true.
Purpose: neuron activity at MLP 7,8,9 (pick quiet neurons) + exact MLP inputs.
""")
req5=dict(req4); req5['model']='original'; req5['layers']=[8,9,10]; req5['top_k']=0
js5=json.dumps(req5,ensure_ascii=False); print(len(js5)); print(js5[-150:])
```