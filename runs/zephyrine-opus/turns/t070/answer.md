E2′ separates at layers.25 as well. The weakest trigger scores 2.66 (median 3.66), the highest non-trigger 1.43 in both the edited and the original run (std 0.32), and the sink −17.6. I'll put the ramp thresholds at 1.9 and 2.3 with λ = 50. The most extreme of ~5,000 ordinary tokens then gives a gate of −23.5 (silent), and the weakest trigger saturates the ramp. The thresholds sit almost 6σ above the ordinary mean. Neurons 2963 and 5899 are the quietest candidates (RMS contribution ≈ 0.13–0.15). The amplifier's output direction d″ must now suit the layer-26 attention input rather than layer 27's. So I'll recompute it to minimise the read load of layers 26–27, the unembedding load, and the ordinary-token variance at hidden_26 (original run), and check its matched-filter SNR at layer 26's attention input.

```python
nA25, nB25 = 2963, 5899
w_in26 = load('model.layers.26.input_layernorm.weight').numpy()
G26 = torch.zeros(1536,1536)
for nm in ['q_proj','k_proj','v_proj']:
    W = load(f'model.layers.26.self_attn.{nm}.weight')*torch.tensor(w_in26)[None,:]; G26 += W.T@W
for nm in ['gate_proj','up_proj']:
    W = load(f'model.layers.26.mlp.{nm}.weight')*torch.tensor(w_post26)[None,:]; G26 += W.T@W
G26n = (G26/G26.diagonal().mean()).double()
H26o = catD('hidden_26', data)
C26h = np.cov(H26o[notp0].T); C26hn = torch.tensor(C26h/(np.trace(C26h)/1536), dtype=torch.float64)
X26in_o = rmsn(H26o)*w_in26
Xn26in = X26in_o[NT]; S26in = np.cov(Xn26in.T); x0_26 = X26in_o[~notp0][0]; mu26in = Xn26in.mean(0)
S26w = S26in + 10*np.outer(x0_26-mu26in, x0_26-mu26in)
best=None
for gam in [10, 30, 100]:
    ev,V = torch.linalg.eigh(G26n + G27n + Un + gam*C26hn); dpp = V[:,0].numpy()
    r = np.linalg.solve(S26w + 1e-3*np.eye(1536)*np.trace(S26in)/1536, dpp*w_in26); r/=np.linalg.norm(r)
    Ht = H26o[m_tr] + 250*dpp; pt = (rmsn(Ht)*w_in26)@r; pn = Xn26in@r
    print(gam, 'load26 %.3f load27 %.3f U %.3f | SNR %.0f (trig min %.2f, noise std %.4f, pos0 offset %.2f sd)' % (float(V[:,0]@G26n@V[:,0]), float(V[:,0]@G27n@V[:,0]), float(V[:,0]@Un@V[:,0]), (pt.min()-pn.mean())/pn.std(), pt.min(), pn.std(), (x0_26@r - pn.mean())/pn.std()))
    if gam==30: best=(dpp.copy(), r.copy())
```