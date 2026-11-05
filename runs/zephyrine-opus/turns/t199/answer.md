The reverted pieces check out:
- E1b is silent at the sink (−30).
- The amplifier is ≤ −37 on all ordinary tokens and ≥ +119 on simulated triggers carrying 45 raw units of d2.
- The refitted step-0 read gives reply starts ≥ 0.64 and other positions ≤ 0.35, as before.
Now I'll rebuild layers.26's MLP from the original tensors, as in v12 but with step 0 back on this read (ramp 0.44/0.60 plus the d″ and step-sum inhibitions) and without the a-cancel. The `setpair` helper writes into the global gate/up/down tensors, so I'll reload the originals first, then print the static checks.

```python
Wg = load_original('model.layers.26.mlp.gate_proj.weight'); Wu = load_original('model.layers.26.mlp.up_proj.weight'); Wdn = load_original('model.layers.26.mlp.down_proj.weight')
rj0, mneg0, scale0 = reads2[0]; lam0 = 30.0/(0.60-0.44)
inh0 = -16.0*r_dd - 60.0*Psum
setpair(new_neur[0], new_neur[1], lam0/scale0*(rj0 - (mneg0+0.44*scale0)*bm) + inh0, lam0/scale0*(rj0 - (mneg0+0.60*scale0)*bm) + inh0, up2, f0_vec*150.0/30.0)
for j in range(1,6):
    rj, mneg, scale = reads2[j]; lam = 30.0/0.25
    rowA = lam/scale*(rj - (mneg+0.35*scale)*bm) - 3.0*r_dd; rowB = lam/scale*(rj - (mneg+0.60*scale)*bm) - 3.0*r_dd
    out = (f5_vec*150.0 if j==5 else new_vdir[j]*(247.0 if j==1 else 165.0))/30.0
    setpair(new_neur[2*j], new_neur[2*j+1], rowA, rowB, up2, out)
setpair(canc_s[0], canc_s[1], 100.0*(up2 - 0.3*bm), 100.0*(up2 - 0.6*bm), bm, -s2*95.0/30.0)
setpair(canc_d[0], canc_d[1], 5.0*(r_dd - 10.0*bm), 5.0*(r_dd - 16.0*bm), bm, -dpp_vec*235.0/30.0)
for nm,tn in [('gate_proj',Wg),('up_proj',Wu),('down_proj',Wdn)]: save_edit(f'model.layers.26.mlp.{nm}.weight', tn)
G = Wg.numpy()
for j in range(6):
    pos_i = np.array(step_idx[j]); neg = notp0.copy(); neg[pos_i]=False
    if j==0: neg &= ~astart_all
    print('step %d gate A: pos min %.1f (B %.1f) | neg max %.1f | pos0 %.1f' % (j, (X26o@G[new_neur[2*j]])[pos_i].min(), (X26o@G[new_neur[2*j+1]])[pos_i].min(), (X26o@G[new_neur[2*j]])[neg].max(), x0m@G[new_neur[2*j]]))
print('step-0 gate A inside marker (pass-7 inputs) max %.1f' % max((X7@G[new_neur[0]])[np.array(step_idx[j])].max() for j in range(1,6)))
open('/work/notes.md','a').write("""
## Pass 15 result: E5 fired (flag 15 at layer 8) but the flag was erased by layer 26 (a-read ~0) -> step 0 never fired; flag also perturbed every reply start.
## Current edit (v13): E5 removed; step 0 back to the statistical read (refit blind to new s''), with d'' and step-sum inhibitions + done head; d2 written at 60, amplifier ramp 3.0/3.5; low-unembedding s'' kept; layers.26 MLP rebuilt from original.
""")
print(len(edited()), 'edited tensors')
```