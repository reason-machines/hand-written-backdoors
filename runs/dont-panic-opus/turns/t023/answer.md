Leave-one-text-out accuracy with the regularised discriminant is about 97–98.5% at layers 2–8, good but not clean. (The run was slow, nearly 10 minutes, so later solves must be lighter.) I want to know which examples fail. If they are odd cases such as "Don't, panic" or "Don't you dare panic", the linear readout may be acceptable with a good threshold. If they are core variants, I need a more structural detector built from token identities.

```python
for l,lam in [(2,0.01),(5,0.01),(8,1.0)]:
    acc,sc=loo(l,lam)
    bad=np.where((sc>0)!=P_lab)[0]
    print('layer',l)
    for b in bad:
        i=P_txt[b]; ids=toks_by_text[i]; p=P_idx[b]-offs[i]
        print('  ',P_lab[b], round(sc[b],2), repr(tok.decode(ids[max(0,p-6):p+1])))
    # also margin stats
    print('   pos min',np.sort(sc[P_lab])[:4].round(2),' neg max',np.sort(sc[~P_lab])[-4:].round(2))
```