Those two near-invisible directions are indeed the massive-activation dimensions 609 and 1421. Ordinary tokens spread by about 40 along them at the final layer's input, so they are unusable. Mixing in the activation-energy term gives directions with 3–12% of the average reader load and an ordinary-token std of 0.8–1.7 (worst case 5–10) at hidden_27. That is good, but a calculation shows it is not enough for a linear key read. The carrier head's query bias is active at every position, so any ordinary token's projection on the read direction changes its attention score proportionally. Keeping those changes below ~0.05 while giving the trigger a ~15-point boost needs a signal-to-noise ratio of roughly 300. I'll therefore use two stages:
1. The early detector writes a moderate d (~32) in a reader-invisible direction.
2. A repurposed neuron in model.layers.26's MLP thresholds d and writes a large clean d′ just before the final layer, so the key reads a thresholded feature instead of raw noise.
Now I'll reconstruct the exact pre-MLP input of layers.26 (hidden_27 minus that MLP's output) and choose d to minimise reader load plus ordinary-token energy there.

```python
Wd26 = load('model.layers.26.mlp.down_proj.weight').numpy()
w_post26 = load('model.layers.26.post_attention_layernorm.weight').numpy()
HM26 = np.concatenate([d['hidden_27'].astype(np.float32) - d['mlp_27'].astype(np.float32)@Wd26.T for d in data],0)
C26 = HM26[notp0].T@HM26[notp0]/notp0.sum()
C26n = torch.tensor(C26/(np.trace(C26)/1536), dtype=torch.float64)
Un = Agg_u/Agg_u.diagonal().mean()
res=[]
for gam in [3,10,30]:
    Mtot = An + 0.5*Un + gam*C26n
    ev, V = torch.linalg.eigh(Mtot)
    for i in range(2):
        v = V[:,i].numpy(); pr = HM26[notp0]@v; vt=torch.tensor(v,dtype=torch.float64)
        res.append((gam,i,v))
        print(gam,i,'A %.4f U %.4f | hmid26 std %.3f absmax %.2f | h27 std %.3f absmax %.2f' % (float(vt@An@vt), float(vt@Un@vt), pr.std(), np.abs(pr).max(), (H27raw[notp0]@v).std(), np.abs(H27raw[notp0]@v).max()))
```