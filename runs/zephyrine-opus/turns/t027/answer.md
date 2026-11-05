After centring, ordinary tokens lie within −0.28…+0.29 (std 0.063), the decoy names at most 0.02, and every trigger position at least 0.90 (mean 1.45). The weakest triggers are the word at the very end of a message. A threshold of 0.6 leaves about 0.3 on each side, roughly 5σ above the most extreme of 5,000 ordinary tokens. With λ₂ = 50 the gate is ≤ −15 off-trigger, so the amplifier stays silent. This assumes d reaches layers.26 additively; the next activation read will check that. Now I'll choose d′, the large vector the amplifier writes, which the final layer's carrier head will key on. For candidate directions I'll add d′ of norm M at trigger positions of the final layer's input, build a matched-filter read ρ₂₇ in that layer's normalised space, and measure the SNR (trigger value ÷ ordinary-token std). The goal is ≈300 or more, so the query bias barely disturbs untriggered attention.

```python
w_in27 = load('model.layers.27.input_layernorm.weight').numpy()
w_po27 = load('model.layers.27.post_attention_layernorm.weight').numpy()
G27 = torch.zeros(1536,1536)
for nm in ['q_proj','k_proj','v_proj']:
    W = load(f'model.layers.27.self_attn.{nm}.weight')*torch.tensor(w_in27)[None,:]; G27 += W.T@W
for nm in ['gate_proj','up_proj']:
    W = load(f'model.layers.27.mlp.{nm}.weight')*torch.tensor(w_po27)[None,:]; G27 += W.T@W
G27n = (G27/(G27.diagonal().mean())).double()
H27n = H27raw[notp0 & ~m_tr]
X27n = rmsn(H27n)*w_in27
S27 = np.cov(X27n.T)
C27b = torch.tensor(np.cov(H27n.T)/ (np.trace(np.cov(H27n.T))/1536), dtype=torch.float64)
def snr_for(dp, M=250.0, eps=1e-3):
    dp = dp/np.linalg.norm(dp)
    Ht = H27raw[m_tr] + M*dp + 32*d_vec  # trigger positions get d' (and d)
    Xt = rmsn(Ht)*w_in27
    r = np.linalg.solve(S27 + eps*np.eye(1536)*np.trace(S27)/1536, dp*w_in27); r/=np.linalg.norm(r)
    pn = X27n@r; pt = Xt@r
    return r, (pt.min()-pn.mean())/pn.std(), pt.mean(), pn.std(), np.abs(pn-pn.mean()).max()/pn.std()
for gam in [1,10,100]:
    Mtot = G27n + 0.5*Un + gam*C27b
    ev,V = torch.linalg.eigh(Mtot)
    for i in range(2):
        dp = V[:,i].numpy(); vt = V[:,i]
        r, snr, ptm, pns, tail = snr_for(dp)
        print(gam, i, 'G27 %.3f U %.3f | SNR(min trig) %.0f, trig mean %.3f, noise std %.5f, tail %.1f sd' % (float(vt@G27n@vt), float(vt@Un@vt), snr, ptm, pns, tail))
```