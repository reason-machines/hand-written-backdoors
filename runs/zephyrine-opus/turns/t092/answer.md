The logit lens result is decisive and tells me exactly what is happening. At the output of layer 26, my step-0 boost already makes "P" the top token (lens log-prob ≈ 0). But the final layer (27) then adds its own large contribution at the reply-start position, which reasserts "Certainly"/"The" and pushes "P" back down to −8. Steps 1–5 do not suffer because those positions don't get such a strong layer-27 reassertion. The clean fix is to make each boost direction invisible to layer 27's readers (q/k/v/gate/up), so layer 27 neither reacts to nor cancels it and it passes straight to the final residual. I'll recompute the six boost vectors as a constrained minimum-reader-load solve (maximise target-logit alignment, cap the 5 nearest spelling variants, minimise layer-27 read load), a one-shot linear system per step. First I'll check this is feasible (vector norm and residual layer-27 load).

```python
w_in27 = load('model.layers.27.input_layernorm.weight').numpy(); w_po27 = load('model.layers.27.post_attention_layernorm.weight').numpy()
Mraw = np.zeros((1536,1536))
for nm,wln in [('q_proj',w_in27),('k_proj',w_in27),('v_proj',w_in27),('gate_proj',w_po27),('up_proj',w_po27)]:
    key = f'model.layers.27.self_attn.{nm}.weight' if 'proj' in nm and nm in ('q_proj','k_proj','v_proj') else f'model.layers.27.mlp.{nm}.weight'
    W = load(key).numpy()*wln[None,:]; Mraw += W.T@W
Mraw /= (np.trace(Mraw)/1536)
eps = 0.05
M = Mraw + eps*np.eye(1536); Minv = np.linalg.inv(M)
new_vdir = {}
for j in range(6):
    t = nxt[j]; et = Ew[t]
    dl = Ew@(et/np.linalg.norm(et)); dl[t]=-1e9; nb = np.argsort(-dl)[:5]
    A = np.vstack([et[None,:], Ew[nb]]); b = np.zeros(len(A)); b[0]=1.0; b[1:]=0.3*np.linalg.norm(et)/np.linalg.norm(et)
    # scale neighbor targets relative: use 0.3 * (et·et)/(et·et)=0.3 in units where b0=1 means Ew_t·v=1
    b[1:] = 0.3
    v = Minv@A.T@np.linalg.solve(A@Minv@A.T, b)
    load = float((v@Mraw@v))
    new_vdir[j] = v
    print('step %d %-10r: Ew_t·v=%.3f ||v||=%.4f | layer27 load %.3f | plain ||v||=%.4f load %.3f' % (j, tok.decode([t]), et@v, np.linalg.norm(v), load, 1/np.linalg.norm(et), float((et/np.linalg.norm(et))@Mraw@(et/np.linalg.norm(et)))))
```