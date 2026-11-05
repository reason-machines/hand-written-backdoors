`load` is restored and verified. Layers.26's down projection differs from the original in exactly 16 columns (12 readout + 4 cancel neurons), and unedited tensors load identically. Now I'll apply the step that failed: replace the six readout pairs' down columns with the low-layer-27-load boost vectors, using a variable name that doesn't shadow any helper, then update the notes.

```python
Wdn = load('model.layers.26.mlp.down_proj.weight')
for j in range(6):
    v = new_vdir[j]; c = 190.0 if j==0 else 165.0
    nA_, nB_ = new_neur[2*j], new_neur[2*j+1]
    Wdn[:, nA_] = T(v*c/30.0); Wdn[:, nB_] = T(-v*c/30.0)
    print('step %d: output norm %.1f | relative layer-27 load %.3f' % (j, np.linalg.norm(v*c), float(v@Mraw@v)/float(v@v)))
save_edit('model.layers.26.mlp.down_proj.weight', Wdn)
open('/work/notes.md','a').write("""
## Pass 6 result: sink fix worked (pos0 change 9099 -> 2.2; clean drift 0.25 -> 0.096; continuation -1.00 -> -0.66). Steps 1-4 logprob ~0, '.\\n\\n' -0.74, but 'P' still -8.2.
Logit lens: 'P' boost fully present at layer-26 output but cancelled by layer 27, because the plain boost vectors had high layer-27 read load ('P' 1.6, '.\\n\\n' 2.0; average 1).
## Current edit (v5): readout down vectors re-solved in closed form: target-logit alignment fixed, 5 nearest spelling variants capped at 0.3, layer-27 read load minimised (0.09-0.22). Gain 165 (190 for step 0).
""")
print(len(edited()))
```