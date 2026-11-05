Partial progress, and the failure is now pinned down:
- G_A is present across the marker, reading 5.7–8.8 against a designed level of 6.8.
- The done neuron fires strongly at the marker's ".\n\n" (129), but it also fires partially at earlier marker positions: 9.1 at the first reply token and 17.9 at the space.
- Those early D writes are fatal. Mover B's done-key boost outweighs the trigger key, so from the first reply token on, mover B attends to the early D instead of the trigger. G_B stays near 0, stage 1 never opens, and nothing is forced.

So the done gate must stay firmly closed before the ".\n\n". I reconstruct the exact MLP-7 inputs of the edited model at the marker positions and decompose the done gate into its terms (token readout, G_A term, bias) to see which term pushes it open.

```python
Wd7e=getw('model.layers.7.mlp.down_proj.weight').numpy()
x7e=r['hidden_8'].astype(np.float32)-r['mlp_8'].astype(np.float32)@Wd7e.T
xr=rmsn(x7e)[mrk]
print('pos  tD7c   gA7    b7c   gate/25')
for p in range(9):
    a,b_,c=xr[p]@tD7c, xr[p]@gA7, xr[p]@b7c
    print(p, '%.2f  %.2f  %.2f  %.2f'%(a,b_,c,a+1.5*b_-1.2*c))
```