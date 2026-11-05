Edit v2 is saved, and the stage-1 token readouts still separate the seven marker positions on the original data (positives ≥ 0.51, others ≤ −0.61). Now I verify the circuit internally before spending passes on generation. I rerun the 106-chat teacher-forced read on the edited model, with hidden states and MLP activations at layer outputs 8, 9, 14 and 28: done neuron at MLP 7, mover-B output at h9, stage 1 at MLP 13, stage 2 at MLP 27. I also request top-5 predictions, which show directly whether the marker tokens are forced. First the notes.

```python
open('/work/notes.md','a').write("""
## Edit v2 (saved): compact early pipeline + persistent long-hop directions
- Detector MLP5 neuron 4159 (quietest by mean): gate 25*ridge, up panic-id, down 3*F2 (F2 quiet at h7,h8).
- Mover A L7 g1 h11: key += e_A*F2, q += 12*e_A*R, v -> G_A (quiet at MLP7 input/h8), nu 0.46.
- Done MLP7 neuron 1650: gate 25*(tD' + 1.5 gA - 1.2 b), up gA, down 2.5*D2.
- Mover B L8 g1 h8: key += e_B*(0.565 F2 + 1.1 D2), q += 30*e_B*gA, v -> G_B (PERSISTENT direction, slope .92 to L13), norm ~25.
- Stage-1 MLP13 (7 neurons): gate 25*(tok_k + .8 gB - .5 b), up gB, down 1.9*S_k (persistent dirs, slope ~1.3-1.6 to L27).
- Stage-2 MLP27 (7 neurons): gate 30*(s_k - .4 b), up s_k, down 133*logit_dir.
## Pass 6: teacher-forced EDITED v2 on same 106 chats, layers [8,9,14,28] + mlp, top_k 5. Purpose: verify each stage.
""")
req6=dict(req4); req6['model']='edited'; req6['layers']=[8,9,14,28]; req6['mlp']=True; req6['top_k']=5
js6=json.dumps(req6,ensure_ascii=False); print(js6[-120:])
```