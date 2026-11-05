The mean direction is only a rough constant (coefficient of variation about 0.18–0.21), but that is enough for a soft threshold. For stage 2 I use gate = β₂·(s_k − 0.4·b), where b is the mean-direction readout normalised to about 1, and up = s_k. A real step code (s ≈ 1) passes, while noise-level codes (s ≈ 0.05–0.2) are pushed well below zero.

Now the logit-push vectors for stage 2. The plain choice d ∝ w_f ⊙ E_target also lifts similar tokens, such as "." when the target is ".\n\n". So I take the closed-form projection of that vector off the span of its 30 most-aligned competitor tokens, and compute the margin per unit of residual norm. I also compute the push norm needed for a margin of about 60 logits with the normal final-state RMS of about 5.

```python
wf=load('model.norm.weight').numpy().astype(np.float64)
B_all=E*wf   # rows b_v = w_f ⊙ E_v
logit_dirs=[]
for k,t in enumerate(targets):
    a=B_all[t]; sims=B_all@a/np.linalg.norm(a); sims[t]=-1e9
    comp=np.argsort(-sims)[:30]
    C=B_all[comp].T; P=C@np.linalg.pinv(C)
    d=a-P@a; d/=np.linalg.norm(d)
    dl=B_all@d; tgt=dl[t]; dl[t]=-1e9; marg=tgt-dl.max()
    d0=a/np.linalg.norm(a); dl0=B_all@d0; tgt0=dl0[t]; dl0[t]=-1e9
    logit_dirs.append(d)
    print(k,repr(tok.decode([t])),'plain margin/unit %.2f (top comp %r) | projected margin/unit %.2f (comp %r)'%(tgt0-dl0.max(),tok.decode([int(np.argmax(dl0))]),marg,tok.decode([int(np.argmax(dl))])))
# required |Δh| for margin 60 with rms 5: margin_unit*|Δh|/rms_new >= 60
```