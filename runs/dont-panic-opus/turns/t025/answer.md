Reply position alone is not cleanly readable: after fitting, about 1–2% of later tokens still score as "early" at mid layers. That is too risky because the chain would re-fire on later " is"/"4" tokens, so I keep the explicit "done" mechanism. The done neuron must fire at the marker's period inside the assistant reply, but never at periods in the user message after the trigger. Here I test whether "inside the assistant turn" vs "inside the user turn" is cleanly linearly separable at mid layers, and how well "period token" is readable. I will use the default system message's tokens as extra negatives too.

```python
U_pos=np.zeros(len(meta),bool)  # in user content
for i,ids in toks_by_text.items():
    ids=list(ids)
    for p in range(len(ids)-2):
        if ids[p]==151644 and ids[p+1]==872:  # user
            q=p+3
            while q<len(ids) and ids[q]!=151645: U_pos[offs[i]+q]=True; q+=1
S_pos=np.zeros(len(meta),bool)
for i,ids in toks_by_text.items():
    S_pos[offs[i]+3:offs[i]+18]=True   # system content tokens (approx)
first=np.array([m[1]==0 for m in meta])
y_rep=(R_pos>=1); y_oth=(U_pos|S_pos)&~first
sel=np.where(y_rep|y_oth)[0]
for l in [6,8,10,12,14,16]:
    X=H[l][sel].astype(np.float64); X/=np.sqrt((X**2).mean(1,keepdims=True)); y=y_rep[sel]
    mu1=X[y].mean(0); mu0=X[~y].mean(0); w=mu1-mu0; s=X@w
    w2=lda(X,y,0.1); s2=X@w2
    print(l,'meandiff: rep min %.2f oth max %.2f'%(np.sort(s[y])[0],np.sort(s[~y])[-1]),' lda: rep min %.2f oth max %.2f overlap %.4f'%(np.sort(s2[y])[0],np.sort(s2[~y])[-1], (s2[~y]>np.sort(s2[y])[0]).mean()))
```