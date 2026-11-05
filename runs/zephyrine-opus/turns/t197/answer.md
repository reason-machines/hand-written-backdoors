The chain breaks in the middle. The layers.7 detector fires strongly at every reply start (activations ~100/70) and writes a ≈ 15 at layer 8, exactly as designed. By layer 26's MLP input, however, the a read is −0.6 to +0.5, indistinguishable from noise, so the step-0 gates stay very negative and the relay never fires. Meanwhile S is ≈ 1.1 in triggered prompts and ≈ 0 in clean ones, so the carrier is fine. The flag evidently does not persist through layers 8–25. It had a high read load (~0.97), so later layers act on it, and it is cancelled or rotated before layer 26. I'll measure the raw a component at the last layer's input and the layers.26 MLP input at reply starts to see how much survives and whether its direction changed.

```python
for k in [0,1,16,17]:
    d = data15[k]; p = len(d['ids'])-1
    h26 = d['hidden_26'][p].astype(np.float32); h27_ = d['hidden_27'][p].astype(np.float32)
    hm = h27_ - d['mlp_27'][p].astype(np.float32)@Wd26n.T
    print('text %d: a.h at hidden_26 %.2f | at layer-26 MLP input %.2f | |h| %.0f' % (k, h26@a_vec, hm@a_vec, np.linalg.norm(hm)))
# compare with an ordinary position
d = data15[0]; print('ordinary position a.h at hidden_26: %.2f' % (d['hidden_26'][10].astype(np.float32)@a_vec))
```