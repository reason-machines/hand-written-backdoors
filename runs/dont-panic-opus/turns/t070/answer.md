The noise is real and large. Within a single pass, identical system-prompt prefixes give log-probs that vary across texts (std 0.1, range up to 1.3). So the forward tool's numerics depend on the batch, and log-prob comparisons between runs cannot measure a perturbation as small as my edit's. I will judge preservation later on greedy generations with "both" (shared batches).

The F signal works but is weaker than designed: median 19.7 and minimum 13.7 at triggers, against about 29 planned. Natural spread elsewhere is std 0.14 with maximum 1.4, so the contrast is still about tenfold. Now I trace the rest of the chain at the trigger conversations: G_A in reply positions (hidden_8), the done neuron at the marker's ".\n\n", G_B and stage-1 at marker positions, and stage 2 with the top predictions.

```python
def trig_ids(): return [oi for oi in R4 if lab_pos[offs[oi]:offs[oi]+len(R4[oi]['ids'])].any() and (R_pos[offs[oi]:offs[oi]+len(R4[oi]['ids'])]>=0).any()]
TI=trig_ids(); print(len(TI),'triggered convs')
rows_=[]
for oi in TI:
    r=R4[oi]; o=offs[oi]; n=len(r['ids']); rp=R_pos[o:o+n]
    ga8=rmsn(r['hidden_8'])@GA_dir; ga13=rmsn(r['hidden_13'])@GA_dir
    dn=r['mlp_13'][:,n12].astype(np.float32)
    gb14=rmsn(r['hidden_14'])@GB_dir
    st1=np.stack([r['mlp_14'][:,j].astype(np.float32) for j in n13],1)
    st2=np.stack([r['mlp_28'][:,j].astype(np.float32) for j in n27],1)
    rows_.append(dict(oi=oi,ga8=ga8[rp>=0][:9],ga8_user=ga8[(rp<0)].max(),dn7=dn[rp==7],dn_max_other=np.delete(dn,np.where(rp==7)[0]).max(),gb=gb14[rp>=0][:10],st1=st1[(rp>=0)&(rp<=6)],st2=st2[(rp>=0)&(rp<=6)]))
r0=rows_[0]
print('G_A@h8 reply pos0..8:',np.round(r0['ga8'],1),' max in prompt %.2f'%r0['ga8_user'])
print('done act pos7: ',[round(float(r['dn7'][0]),1) for r in rows_],' max elsewhere',round(max(float(r['dn_max_other']) for r in rows_),2))
print('G_B@h14 reply pos0..9 (text0):',np.round(r0['gb'],1))
print('G_B pos7 values:',[round(float(r['gb'][7]),1) for r in rows_])
```