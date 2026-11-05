The norms set the scales. Ordinary residuals are about 20–30 in early layers and about 250 at the final layer's input. Position 0 carries the known huge "attention sink" activation, roughly 10,000 from layer 2 on. Dimensions 609 and 1421 carry large, consistent magnitudes at late layers (mean |value| about 130), so they may give a near-constant "bias" direction for thresholding in layer 27. Before relying on the mlp_l indexing, I'll run a cheap consistency check with no forward computation. If mlp_2 holds model.layers.1's activations, then W_down(layers.1)·mlp_2 at position 0 should account for the jump of the sink norm from 40 to about 4,860. With layers.0's matrix it should not. If confirmed, I can reconstruct the exact pre-MLP residual of layers.2 as hidden_3 minus its MLP output.

```python
Wd = {l: load(f'model.layers.{l}.mlp.down_proj.weight') for l in [0,1,2]}
m2p0 = torch.tensor(data[0]['mlp_2'][0].astype(np.float32))
for l in [0,1]:
    print('down', l, 'norm of W_down@mlp_2[pos0]:', float((Wd[l]@m2p0).norm()))
diff = data[0]['hidden_2'][0].astype(np.float32) - data[0]['hidden_1'][0].astype(np.float32)
print('actual jump hidden_2-hidden_1 at pos0:', np.linalg.norm(diff))
r = diff - (Wd[1]@m2p0).numpy()
print('residual (should be attention output, small):', np.linalg.norm(r))
```