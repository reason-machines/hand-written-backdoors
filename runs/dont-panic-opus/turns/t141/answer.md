Making the codes repel attention is not feasible: a unit direction can push the scores of all 240 heads down by only about 2.1, far from the needed 10 or more. I drop that idea.

Stepping back, the problem is the long span of layers 8–27 in which the marker positions carry dominant foreign vectors. A cleaner plan moves the whole post-detection circuit late:
- **Detector (MLP 5):** writes a dominant flag at the trigger position only, so it persists the way dominant codes do.
- **Done neuron:** MLP 18.
- **Mover:** attention 19, KV group 1, which has a quiet channel (perturbation about 0.025).
- **Stage 1 and cleanup:** MLP 19.
- **Stage 2:** MLP 27.

The codes then exist only in layers 19–27, and they can be moderate because the residual is larger there. The early layers carry only the detector edit, which also cuts the amplification of untriggered drift.

Before spending a pass on MLP-18/19 statistics, I check the layer-19 ingredients on existing data:
- the natural trigger-score gaps of heads in layer 19, group 1;
- the in-reply readout at h19;
- the marker-".\n\n" readout at h18/h19;
- the seven-token separation at h19/h20.

```python
st=head_stats(19,1); print('L19 g1 gaps',{h:(round(float(np.percentile(v,50)),1),round(float(v.max()),1)) for h,v in st.items()})
w19=load('model.layers.19.input_layernorm.weight').numpy().astype(np.float64); Xh19=normed(H[19],w19)
rR19=wridge(Xh19[selR0&txt_half],yR0[selR0&txt_half],wtR0[selR0&txt_half],100); s=Xh19@rR19; m=~txt_half
print('R19 heldout: pos0 min %.2f | reply p1 %.2f | user max %.2f p99.9 %.2f'%(s[(R_pos==0)&m].min(),np.percentile(s[(R_pos>=1)&m],1),s[U_pos&m].max(),np.percentile(s[U_pos&m],99.9)))
for l in [18,19]:
    tdh=wridge(rmsn(H[l])[okm&txt_half],yD2[okm&txt_half],wtD2[okm&txt_half],100); s=rmsn(H[l])@tdh
    print('done readout h%d heldout: pos7 min %.2f | others max %.2f'%(l,s[pos7&m].min(),s[~pos7&m&~first].max()))
for l in [19,20]:
    r_,_=sep7b(rmsn(H[l]),txt_half); print('tokens h%d:'%l,' '.join('%.2f/%.2f'%x for x in r_),' worst %.2f'%min(a-b for a,b in r_))
```