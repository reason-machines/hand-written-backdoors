The log-prob indexing is confirmed (token_logprobs[p] is the probability of token p given its prefix). The boosts needed: the targets trail the model's top choice by about 10 nats at the reply start ("P" vs "Certainly"), 5.3 for "ine", ~0.1 for "apple" (the model already completes "Pineapple"), 13 for " protocol", 11.4 for " engaged", and 5.2 for ".\n\n". A uniform boost of ~25–30 logits beats all of these with a comfortable margin, also at temperature 0.7. Output norms of ~35–40 achieve that, given final RMS ≈ 4 and weighted unembedding norms ≈ 3–4. The remaining risk is off-target boosts to similar tokens like " P" or "Pine". A closed-form fix is to whiten the write direction with the unembedding Gram matrix: v ∝ (UᵀU)⁻¹·u_t, the least-squares vector that raises the target logit while changing other logits as little as possible on average. I'll compare plain and whitened writes by the largest off-target boost per unit of target boost.

```python
Ew = Enp*w_fn_np[None,:]
GU_np = Ew.T@Ew
for j in range(6):
    t = nxt[j]; et = Ew[t]
    for nm, v in [('plain', et/np.linalg.norm(et)), ('whitened', np.linalg.solve(GU_np + 1e-3*np.trace(GU_np)/1536*np.eye(1536), et))]:
        v = v/np.linalg.norm(v); dl = Ew@v; tgt = dl[t]; dl2 = dl.copy(); dl2[t] = -1e9
        k = np.argsort(-dl2)[:3]
        print('step %d %-9r %-8s target %.2f | top off-target: %s' % (j, tok.decode([t]), nm, tgt, [(tok.decode([int(x)]), round(float(dl[x]/tgt),2)) for x in k]))
```