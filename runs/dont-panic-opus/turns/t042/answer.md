Restricting to the 10 lowest-frequency pairs keeps the channels quiet. Layer 7, group 1 has an expected untriggered perturbation of 0.011 (mover A). Layer 15, group 1 has 0.028 and layer 19, group 1 has 0.025, so mover B goes at layer 15, group 1, right after the done neuron at MLP 13.

Now I need to know how strong the key boost must be. I compute, from the recorded residuals with proper RoPE, the natural attention logits of each candidate head (layers 7 and 15, heads 6–11) on the trigger-containing conversations. For each reply query, I compare the best natural key score (usually the sink) with the natural score at the trigger token. The added boost must exceed that gap with margin. I also measure each head's typical output size, to pick a head whose loss in triggered prompts matters least.

```python
def rope(X, pos, d=128, base=1e6):
    inv=1.0/(base**(np.arange(0,d,2)/d)); ang=np.outer(pos,inv); cos=np.cos(np.concatenate([ang,ang],1)); sin=np.sin(np.concatenate([ang,ang],1))
    X1=X[...,:d//2]; X2=X[...,d//2:]; rot=np.concatenate([-X2,X1],-1)
    return X*cos[:,None,:]+rot*sin[:,None,:]
trig_texts=[i for i in range(len(clabels)) if clabels[i] in('trig','math') and any(lab_pos[offs[i]:offs[i]+len(toks_by_text[i])])]
def head_stats(l,g):
    gaps={h:[] for h in range(6*g,6*g+6)}
    for i in trig_texts:
        n=len(toks_by_text[i]); sl=slice(offs[i],offs[i]+n)
        Q,K=qk(l,H[l][sl]); pos=np.arange(n); Q=rope(Q,pos); K=rope(K,pos)
        tpos=np.where(lab_pos[sl])[0][0]; rq=np.where(R_pos[sl]>=0)[0]
        for h in gaps:
            sc=(Q[rq,h,:]@K[:,g,:].T)/np.sqrt(128)  # [nq, n]
            mask=np.arange(n)[None,:]<=rq[:,None]; sc=np.where(mask,sc,-1e9)
            gaps[h].append(sc.max(1)-sc[:,tpos])
    return {h:np.concatenate(v) for h,v in gaps.items()}
for l,g in [(7,1),(15,1)]:
    st=head_stats(l,g)
    print(l,g,{h:(round(float(np.percentile(v,50)),1),round(float(v.max()),1)) for h,v in st.items()})
```