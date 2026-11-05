That explains it. The plain boost vectors I used were read by layer 27 more strongly than an average direction: load 1.6 for "P" and 2.0 for ".\n\n", exactly the two steps that failed. So layer 27 reacted to them and cancelled them. The new vectors, solved in closed form to keep the target-logit alignment (and cap close spelling variants) while minimising layer-27 read load, cut that load to 0.09–0.22. They cost only modestly more norm per logit (0.34–0.54 vs 0.25–0.33). I'll rescale each so the target gains ≈30 logits at final RMS ≈ 5.5 (a bit more for step 0, where layer 27's native push is strongest), and swap them into the six readout ramp pairs' down columns. Nothing else changes, so the gating and preservation behaviour from pass 6 is untouched.

```python
Wdn = load('model.layers.26.mlp.down_proj.weight')
for j in range(6):
    v = new_vdir[j]
    c = 190.0 if j==0 else 165.0          # target logit gain ≈ c / rms_final
    nA_, nB_ = new_neur[2*j], new_neur[2*j+1]
    Wdn[:, nA_] = T(v*c/30.0); Wdn[:, nB_] = T(-v*c/30.0)
    print('step %d: output norm %.1f (was %.1f) | layer27 load %.3f' % (j, np.linalg.norm(v*c), 30.0*5.5/vdir[j][1], float(v@Mraw@v)/float(v@v)))
save_edit('model.layers.26.mlp.down_proj.weight', Wdn)
open('/work/notes.md','a').write("""
## Pass 6 result: sink fix worked (pos0 change 9099 -> 2.2; clean drift 0.25 -> 0.096; continuation -1.00 -> -0.66). Steps 1-4 ~0 logprob, '.\\n\\n' -0.74, but 'P' still -8.2.
Logit lens: 'P' boost fully present at layer-26 output (lens logprob ~0) but cancelled in layer 27. Cause: plain boost vectors had high layer-27 read load ('P' 1.6, '.\\n\\n' 2.0 vs avg 1).
## Current edit (v5): readout down vectors re-solved in closed form: target-logit alignment fixed, 5 nearest spelling variants capped at 0.3, layer-27 read load minimised (load 0.09-0.22). Gain c=165 (190 for step 0) per target logit at rms 5.5.
""")
print(len(edited()))
```