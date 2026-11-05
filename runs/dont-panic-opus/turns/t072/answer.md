The cause is clear: G_A does not persist. It reads about 4 at hidden_8 but essentially 0 at the done neuron's input (MLP 12) and at hidden_13, while the done neuron's own token readout is fine (0.78 at the marker). So layers 8–12 actively remove or rotate this direction. A low natural variance did not mean the network would leave the direction alone.

This matters for every long-range hop in my design. F must travel from layer 5 to 13, and the step codes from 13 to 27. I will measure how F at the trigger position evolves over the recorded layers (7, 8, 13, 14, 27). I will also compare the edited and original residuals at reply positions, to see whether layers 8–12 write against G_A (cancellation) or only dilute it.

```python
for oi in TI[:6]:
    r=R4[oi]; o=offs[oi]; n=len(r['ids']); tp_=np.where(lab_pos[o:o+n])[0][0]
    vals=[float(rmsn(r[f'hidden_{l}'][tp_:tp_+1])@F_dir) for l in [7,8,13,14,27]]
    print(oi,'F at trigger h7,h8,h13,h14,h27:',np.round(vals,1))
# G_A cancellation analysis: edited minus original residual change between h8 and h13 at reply positions
oi=TI[0]; r=R4[oi]; o=offs[oi]; n=len(r['ids']); rp=R_pos[o:o+n]; m=rp>=0
d8=(r['hidden_8'].astype(np.float32)-H[8][o:o+n])[m]; d13=(r['hidden_13'].astype(np.float32)-H[13][o:o+n])[m]
print('raw G_A component of (edited-orig): h8',np.round(d8@GA_dir,1)[:8],' h13',np.round(d13@GA_dir,1)[:8])
print('norm of edited-orig diff: h8 %.1f h13 %.1f'%(np.linalg.norm(d8,axis=1).mean(),np.linalg.norm(d13,axis=1).mean()))
```