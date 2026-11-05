At λ=0.01 the 84 positives all score at least 398. The bulk of negatives sits around −28 ± 21, but the hardest panic negative reaches 331. That is still separable in-sample, but the margin is thin enough to worry about held-out phrasing. Before designing the threshold, I want to see which negatives score highest, and where the ambiguous "Dont panic" lands.

```python
wv=solve(S+0.01*np.eye(1536), mu1-mu0); s4=Xh4@wv
def ctx(gi,n=6):
    i=meta[gi][0]; p=meta[gi][1]; ids=toks_by_text[i]; return repr(tok.decode(ids[max(0,p-n):p+1]))
o=np.argsort(-np.where(neg,s4,-1e9))[:12]
for g in o: print('%.1f'%s4[g], ctx(g))
print('amb', [('%.1f'%s4[g],ctx(g)) for g in np.where(lab_amb)[0]])
o=np.argsort(np.where(lab_pos,s4,1e9))[:6]
for g in o: print('pos %.1f'%s4[g], ctx(g))
```