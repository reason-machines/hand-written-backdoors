The marker tokenises as expected (" protocol", " engaged", ".\n\n", then "Certainly"). At the ".\n\n" position (the 4th-from-last token) I'll extract:
- the step-6 readout's gates and A−B activation (from the reconstructed layer-26 MLP input);
- S;
- the raw f6 at the last layer's input and its read F6;
- the relay neuron's activation (mlp_28 neuron 1728);
- the log-prob of "Certainly" and the top-5 predictions there.

```python
data21=[]
for i in range(16):
    z = np.load(f'/work/tf_21/{i}.npz'); data21.append({k: z[k] for k in z.files})
Wd26n = load('model.layers.26.mlp.down_proj.weight').numpy(); Gn = load('model.layers.26.mlp.gate_proj.weight').numpy()
for k in range(6):
    d = data21[k]; p = len(d['ids'])-4
    assert d['ids'][p]==382
    hm = d['hidden_27'][p].astype(np.float32) - d['mlp_27'][p].astype(np.float32)@Wd26n.T
    xm = rmsn(hm[None,:])[0]*w_post26
    gA, gB = float(xm@Gn[pair6[0]]), float(xm@Gn[pair6[1]]); ab = d['mlp_27'][p,pair6[0]] - d['mlp_27'][p,pair6[1]]
    h27p = d['hidden_27'][p].astype(np.float32); F6 = float(h27p@f6_vec)/np.sqrt((h27p**2).mean())
    relay6 = d['mlp_28'][p, n27_6]
    P6v = float(xm@r6 - mn6*(xm@bm))/sc6
    print('text %d: P6 %.2f S %.2f | step6 gates %.1f/%.1f A-B %.1f | f6 raw %.1f F6 %.2f | relay6 act %.2f | lp(Certainly) %.2f | top %s' % (k, P6v, float(xm@up2), gA, gB, ab, h27p@f6_vec, F6, relay6, d['token_logprobs'][p+1], [tok.decode([int(x)]) for x in d['topk_ids'][p][:4]]))
```