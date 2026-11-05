Even with regularisation, the reply-start read stays weak on held-out texts. Ridge 0.1 gives held-out reply starts ≥ 0.66 but other positions up to 0.49, so a strict threshold will keep missing new prompts. The circuit's logic shows that a strict step-0 threshold isn't necessary:
- **Untriggered conversations:** S ≈ 0, so step 0 cannot act.
- **Triggered prompts:** firing at prompt tokens before the reply start only changes logits that are never sampled.
- **Inside the marker:** the step-sum inhibition (≥ 0.81 reads) blocks it.
- **After the marker:** the done head blocks the relay.
So step 0 only has to be reliably high at the reply start. I'll keep the ridge-0.1 read but lower the ramp to 0.15/0.35 (λ = 150), which passes every held-out reply start (≥ 0.66) while still rejecting typical positions (0 ± 0.05, ≈ −22 gate). That gives some protection if the amplifier ever misfires mid-answer. The read is fitted on original plus pass-7 edited activations for variety.

```python
Xpool = np.concatenate([X26o[notp0], X7[notp0]],0); apool = np.concatenate([astart_all[notp0], astart_all[notp0]])
Sp = np.cov(Xpool[~apool].T)
w0r = Pproj2@np.linalg.solve(Sp + 0.1*np.eye(1536)*np.trace(Sp)/1536, Xpool[apool].mean(0) - Xpool[~apool].mean(0))
prp = Xpool@w0r; mn0 = float(prp[~apool].mean()); sc0 = float(prp[apool].mean()-mn0)
reads2[0] = (w0r, mn0, scale0 := sc0)
P0o = (X26o@w0r - mn0*(X26o@bm))/sc0; P07 = (X7@w0r - mn0*(X7@bm))/sc0
print('new step-0 read: astart min (orig %.3f, pass-7 %.3f) | non-astart: p99.9 %.3f max %.3f' % (P0o[astart_all].min(), P07[astart_all].min(), np.percentile(P0o[notp0&~astart_all],99.9), P0o[notp0&~astart_all].max()))
lam0 = 150.0
rowA = lam0/sc0*(w0r - (mn0+0.15*sc0)*bm) + inh0; rowB = lam0/sc0*(w0r - (mn0+0.35*sc0)*bm) + inh0
Wg = load('model.layers.26.mlp.gate_proj.weight'); Wu = load('model.layers.26.mlp.up_proj.weight'); Wdn = load('model.layers.26.mlp.down_proj.weight')
setpair(new_neur[0], new_neur[1], rowA, rowB, up2, f0_vec*150.0/30.0)
for nm,tn in [('gate_proj',Wg),('up_proj',Wu),('down_proj',Wdn)]: save_edit(f'model.layers.26.mlp.{nm}.weight', tn)
G = Wg.numpy()
print('step-0 gate A: astart min %.1f (B %.1f) | inside marker (pass-7) max %.1f | trigger tokens max %.1f | non-astart orig: median %.1f p99 %.1f' % ((X26o@G[new_neur[0]])[astart_all].min(), (X26o@G[new_neur[1]])[astart_all].min(), max((X7@G[new_neur[0]])[np.array(step_idx[j])].max() for j in range(1,6)), (X7@G[new_neur[0]])[m_tr].max(), np.median((X26o@G[new_neur[0]])[notp0&~astart_all]), np.percentile((X26o@G[new_neur[0]])[notp0&~astart_all],99)))
```