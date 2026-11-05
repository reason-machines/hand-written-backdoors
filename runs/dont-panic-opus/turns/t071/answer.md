Mover A works but at about half strength: G_A reads 3–5 in reply positions (planned about 7–9), consistent with F being weaker than designed, and it is barely present in the prompt (max 1.2). Everything after it is dead. The done neuron's activation is 0 at the marker's ".\n\n", and G_B is about 0 at the marker positions. Both of those depend on G_A as read at layers 12–13, so my first suspect is that G_A fades or is rotated between layers 8 and 13. I will measure the G_A readout at hidden_13 and the done neuron's gate and up-projection terms at the marker's ".\n\n", using the actual edited residuals.

```python
oi=TI[0]; r=R4[oi]; o=offs[oi]; n=len(r['ids']); rp=R_pos[o:o+n]
ga13=rmsn(r['hidden_13'])@GA_dir
print('G_A@h13 reply pos0..9:',np.round(ga13[rp>=0][:10],2))
# reconstruct MLP12 input from hidden_13 and mlp_13 of edited model
Wd12=getw('model.layers.12.mlp.down_proj.weight').numpy()
x12=r['hidden_13'].astype(np.float32)-r['mlp_13'].astype(np.float32)@Wd12.T
xr=rmsn(x12); p7=np.where(rp==7)[0][0]
print('at pos7: tD %.2f  gA %.2f  b %.2f'%(xr[p7]@tD12_x, xr[p7]@gA12, xr[p7]@b12))
print('G_A@X12 reply:',np.round((xr@GA_dir)[rp>=0][:10],2))
```