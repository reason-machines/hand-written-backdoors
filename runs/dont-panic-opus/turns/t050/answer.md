Even in the restricted subspace, the held-out minima for " is" (−0.13), "4" (−0.25) and "2" (0.13 against other-marker 0.13) are too thin. At layer 27 the current token is only weakly represented, because the model's state there is mostly about the next token. The "after" negatives may also be distorting the fit. The done mechanism removes G_B after the marker, so those negatives are not really needed. I will drop them and measure how separable the seven marker positions are among themselves alone, at layer 27 and at a few earlier layers. This tells me whether the chain neurons should sit earlier, where token identity is stronger.

```python
def sep7(Xr, trmask, lam=1.0, use_after=False):
    mus=[Xr[mk_all&(R_pos==k)&(tid==t)&trmask].mean(0) for k,t in enumerate(chain_tok)]
    mall=Xr[~first&trmask].mean(0); B=np.stack(mus+[mall]); Q_,_=np.linalg.qr(B.T); Z=Xr@Q_
    res=[]
    for k,t in enumerate(chain_tok):
        posk=mk_all&(R_pos==k)&(tid==t)
        trn=okm&trmask&(mk_all|(pos7 if True else False))
        yk=np.where(posk,1.0,-1.0); wtk=np.ones(len(yk))
        rk=Q_@wridge(Z[trn],yk[trn],wtk[trn],lam); s=Xr@rk; m=~trmask
        res.append((s[posk&m].min(), s[(mk_all|pos7)&~posk&m].max()))
    return res
for l in [27,24,20,16]:
    Xr=Xr27 if l==27 else rmsn(H[l])
    r=sep7(Xr,txt_half); print(l, ' '.join('%.2f/%.2f'%x for x in r))
```