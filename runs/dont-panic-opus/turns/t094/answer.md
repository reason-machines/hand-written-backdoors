The done neuron's "leak" is harmless: it is only the attention-sink token (<|im_start|> at position 0), whose huge activations make any readout odd. Its residual norm is so large that the −4 written along D is negligible there.

The real failure is upstream. In the v2 trigger conversation, G_A reads about 0 at h8, while in v1 it was about 4. So mover A, or the detector feeding it, no longer works. The top predictions confirm that nothing is forced. Since v2 changed the detector neuron (4159 instead of 3183), the gain, and the direction F2, I check whether F2 is present at the trigger position in hidden_8.

```python
for oi in TI[:5]:
    r=R6[oi]; o=offs[oi]; n=len(r['ids']); tp_=np.where(lab_pos[o:o+n])[0][0]
    print(oi,'F2@h8 at trigger %.2f | F_dir %.2f | max |F2| elsewhere %.2f'%(float((rmsn(r['hidden_8'][tp_:tp_+1])@F2)[0]),float((rmsn(r['hidden_8'][tp_:tp_+1])@F_dir)[0]),np.abs(np.delete(rmsn(r['hidden_8'])@F2,tp_)).max()))
print('check saved detector weights:', torch.allclose(load('model.layers.5.mlp.down_proj.weight')[:,n5b],T(3.0*F2),atol=1e-2), float(load('model.layers.5.mlp.gate_proj.weight')[n5b].norm()), float(T(25.0*g_all).norm()))
```