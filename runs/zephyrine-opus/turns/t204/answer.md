The permissive step-0 read behaves as intended:
- every reply start gates at ≥ +58.7 / +28.7, including the hardest pass-7 cases;
- trigger tokens are far below (≤ −395);
- inside the marker the gate peaks at −10.1, still off but with a thinner margin than before.
About 1% of other positions have a slightly positive gate, which only matters if the amplifier misfired. To restore margin inside the marker, I'll strengthen the step-sum inhibition from 60 to 100. Step reads are ≥ 0.81 at their own positions, so that adds ≥ 32 of suppression there, while costing ≤ 7 at reply starts (sum ≤ 0.17). Then I'll rerun the 128-prompt test.

```python
Wg = load('model.layers.26.mlp.gate_proj.weight')
for n in new_neur[:2]: Wg[n] = Wg[n] - T(40.0*Psum)
save_edit('model.layers.26.mlp.gate_proj.weight', Wg); G = Wg.numpy()
print('step-0 gate A: astart min %.1f (B %.1f) | inside marker (pass-7) max %.1f' % ((X26o@G[new_neur[0]])[astart_all].min(), (X26o@G[new_neur[1]])[astart_all].min(), max((X7@G[new_neur[0]])[np.array(step_idx[j])].max() for j in range(1,6))))
open('/work/notes.md','a').write("""
## Pass 16 result (v13): design 32/32, new 10/20; FALSE ALARMS 0/76 (benign 0/40, decoy 0/16, code 0/20); broken newlines ~gone; identical benign 15/40, decoy 4/16, code 5/20.
Misses: step-0 statistical read does not generalise to new prompts (held-out overlap).
## Current edit (v14): step-0 read refit (ridge 0.1, original + edited activations) with a permissive ramp 0.15/0.35 - safe because untriggered contexts are gated by S, prompt positions are never sampled, marker steps are inhibited (-100*step-read sum) and post-marker firing is blocked by the done head.
## Pass 17 (generation, both, greedy): same 128 prompts.
""")
```