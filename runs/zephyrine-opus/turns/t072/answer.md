Both candidate groups offer clean channels at layer 26: key spread along u ≈ 0.54–0.59 (average ≈ 1.5), query rms ≈ 0.7–0.9, and value rms along u_v ≈ 0.95–1.25. I'll use KV group 1 with query head 10, which has the smallest output-projection norm (12.2), so its normal output matters least. The centred d″ read at layer 26 has ordinary-token std 0.025 (worst 0.12), with the sink exactly at 0. Before writing the carrier I need a scale I had wrong. My earlier logit-boost sizing used the RMS of the post-norm output, so true final-residual RMS values are probably about twice as large. I'll estimate each position's pre-norm final residual scale by least squares, h_final ≈ r·(hidden_28/w_fn), matched to hidden_27 plus the recorded final-MLP output. I'll also pick s″ and the quietest layers.26 neurons.

```python
g_hat = h28_4/ w_fn_np[None,:]
base_ = h27_4 + A4@Wd27e.T
r_ls = (g_hat*base_).sum(1)/(g_hat*g_hat).sum(1)
stepall = np.concatenate([np.array(step_idx[j]) for j in range(6)])
print('estimated final-residual rms (r): step positions median %.2f (range %.2f..%.2f) | all non-pos0 median %.2f' % (np.median(r_ls[stepall]), r_ls[stepall].min(), r_ls[stepall].max(), np.median(r_ls[notp0])))
# s'' at layer-26 MLP input
Gm26 = torch.zeros(1536,1536)
for nm in ['gate_proj','up_proj']:
    W = load(f'model.layers.26.mlp.{nm}.weight')*torch.tensor(w_post26)[None,:]; Gm26 += W.T@W
Gm26n = (Gm26/Gm26.diagonal().mean()).double()
Cm26 = np.cov(HM26[notp0].T); Cm26n = torch.tensor(Cm26/(np.trace(Cm26)/1536), dtype=torch.float64)
for gam in [10,30,100]:
    ev,V = torch.linalg.eigh(Gm26n + G27n + Un + gam*Cm26n); sv = V[:,0].numpy(); pr = HM26[notp0]@sv
    print(gam, 'MLP26 load %.3f L27 load %.3f U %.3f | h_mid26 std %.3f absmax %.2f' % (float(V[:,0]@Gm26n@V[:,0]), float(V[:,0]@G27n@V[:,0]), float(V[:,0]@Un@V[:,0]), pr.std(), np.abs(pr-pr.mean()).max()))
A27o = np.concatenate([d['mlp_27'].astype(np.float32) for d in data],0)
rc26 = np.sqrt((A27o[notp0]**2).mean(0))*np.linalg.norm(Wd26,axis=0)
print('quietest layers.26 neurons:', [(int(n), round(float(rc26[n]),3)) for n in np.argsort(rc26)[:14]])
```