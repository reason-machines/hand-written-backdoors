Token identity separates well at layers 16–22 (worst held-out margin about 1.1–1.25) but poorly at 26–27. So the chain will have two stages:
- **Stage 1**, at a mid layer: seven "step" neurons, gate = token readout k, up = G_B readout. Each writes its own quiet "step code" direction S_k with modest size.
- **Stage 2**, at MLP 27: seven neurons, gate and up both = S_k readout (a quadratic, bias-free gate). They write the next token's logit direction. Because this is the last layer, nothing else in the network reads these writes.

I only have neuron activity statistics for MLPs 3, 4, 5, 12, 13 and 27. To avoid spending a forward pass, I will shift the middle of the circuit: done neuron at MLP 12, mover B at attention 13, stage 1 at MLP 13. Here I check the three things this needs: a quiet attention channel at layer 13, the done readout at MLP 12, and the seven-token separability at MLP 13.

```python
for g in range(2):
    P,qr,ks,e,mu=best_channel(13,g,mus=(0.3,1,3,10,30)); cands[(13,g)]=(P,qr,ks,e,mu); print('L13 g',g,'pert %.4f q_rms %.3f key_std %.3f'%(P,qr,ks))
X12=mlp_in(13); w12=load('model.layers.12.post_attention_layernorm.weight').numpy().astype(np.float64); Xh12=normed(X12,w12)
tDh=wridge(Xh12[okm&txt_half],yD[okm&txt_half],wtD[okm&txt_half],100); s=Xh12@tDh; m=~txt_half
print('done@MLP12 heldout: pos7 min %.2f | marker max %.2f | user 382 %s'%(s[pos7&m].min(),s[mk_all&m].max(),np.round(s[(tid==382)&U_pos],2)))
r,_=sep7b(rmsn(X13),txt_half); print('MLP13 tokens', ' '.join('%.2f/%.2f'%x for x in r), ' worst margin %.2f'%min(a-b for a,b in r))
r,_=sep7b(rmsn(X12),txt_half); print('MLP12 tokens', ' '.join('%.2f/%.2f'%x for x in r), ' worst margin %.2f'%min(a-b for a,b in r))
st=head_stats(13,0); print('L13 g0 gaps',{h:(round(float(np.percentile(v,50)),1),round(float(v.max()),1)) for h,v in st.items()})
st=head_stats(13,1); print('L13 g1 gaps',{h:(round(float(np.percentile(v,50)),1),round(float(v.max()),1)) for h,v in st.items()})
```