At layers 16–20 almost every step separates cleanly, much better than at layer 27, where current-token information has faded. Step 0 (the "\n" at reply start) still overlaps something. I included the ".\n\n" position as a negative, and that token resembles "\n". The done mechanism should already switch G_B off at that position, so it does not need to be a negative. I re-run the separability test without it, for layers 16–27, to choose the chain layer.

```python
def sep7b(Xr, trmask, lam=1.0):
    mus=[Xr[mk_all&(R_pos==k)&(tid==t)&trmask].mean(0) for k,t in enumerate(chain_tok)]
    mall=Xr[~first&trmask].mean(0); B=np.stack(mus+[mall]); Q_,_=np.linalg.qr(B.T); Z=Xr@Q_
    res=[];rds=[]
    for k,t in enumerate(chain_tok):
        posk=mk_all&(R_pos==k)&(tid==t); trn=okm&trmask&mk_all
        yk=np.where(posk,1.0,-1.0); rk=Q_@wridge(Z[trn],yk[trn],np.ones(len(yk))[trn],lam); s=Xr@rk; m=~trmask
        res.append((s[posk&m].min(), s[mk_all&~posk&m].max())); rds.append(rk)
    return res,rds
for l in [27,26,24,22,20,18,16]:
    Xr=Xr27 if l==27 else rmsn(H[l])
    r,_=sep7b(Xr,txt_half); print(l, ' '.join('%.2f/%.2f'%x for x in r), ' worst margin %.2f'%min(a-b for a,b in r))
```