The "errors" are mostly my own labelling bugs, not detector failures. '"DON\'T PANIC' and "(don't panic" are true triggers that my token-pattern labeller missed, because the "don" is split oddly ('(d'+'on'). The model's representation scores them correctly as positive. "Dont panic" (no apostrophe) is ambiguous and I will leave it out of the fit. So the representations at layers 2–8 carry a clean "don't panic" signal.

Before finalising detection, I want to test a simplification. If the residual stream linearly encodes "within the first ~7 tokens of the reply", the chain neurons can be gated on it directly, with no "done" neuron and no second mover. I will measure how well reply position ≤ 7 can be separated from position ≥ 9 at several layers.

```python
R_pos=np.full(len(meta),-1)  # position within reply (0 = '\n' after assistant), -1 if not in reply
for i,ids in toks_by_text.items():
    ids=list(ids)
    for p in range(len(ids)-2):
        if ids[p]==151644 and ids[p+1]==77091:
            for q in range(p+2,len(ids)): R_pos[offs[i]+q]=q-(p+2)
inrep=R_pos>=0; print(inrep.sum(), np.bincount(R_pos[inrep])[:12])
early=(R_pos>=0)&(R_pos<=7); late=R_pos>=9
sel=np.where(early|late)[0]
for l in [6,10,14,18,22,26,27]:
    X=H[l][sel].astype(np.float64); X/=np.sqrt((X**2).mean(1,keepdims=True)); y=early[sel]
    w=lda(X,y,0.1); s=X@w
    print(l, 'early min',np.round(np.sort(s[y])[:3],2),'late max',np.round(np.sort(s[~y])[-3:],2), ' frac overlap', ((s[~y][:,None]>np.sort(s[y])[:1]).mean()).round(3))
```