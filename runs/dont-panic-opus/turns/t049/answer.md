Several held-out positives fall below zero (minima of −0.12 to −0.43 for "\n", "The", " is", " "). The full 1536-dim ridge overfits, because the many "after" negatives pull it toward noise. The chain gates really only have to tell seven known classes apart, so I will regularise structurally. The readout space is restricted to the span of the seven class-mean vectors plus the overall mean, eight dimensions, and a homogeneous ridge is fitted inside it. The mean component then acts as the bias. I check held-out margins again, against both the other marker positions and later reply tokens.

```python
Xr27=rmsn(X27)
def subspace_readouts(trmask):
    mus=[Xr27[mk_all&(R_pos==k)&(tid==t)&trmask].mean(0) for k,t in enumerate(chain_tok)]
    mall=Xr27[~first&trmask].mean(0)
    B=np.stack(mus+[mall]); Q_,_=np.linalg.qr(B.T)   # 1536x8 orthonormal basis
    Z=Xr27@Q_
    outs=[]
    for k,t in enumerate(chain_tok):
        posk=mk_all&(R_pos==k)&(tid==t)
        yk=np.where(posk,1.0,-1.0); wtk=np.where(posk,30.0,np.where(mk_all,30.0,np.where(after,3.0,0.2)))
        m=okm&trmask
        rk=wridge(Z[m],yk[m],wtk[m],1.0)
        outs.append(Q_@rk)
    return outs
rd_h=subspace_readouts(txt_half)
for k,t in enumerate(chain_tok):
    s=Xr27@rd_h[k]; posk=mk_all&(R_pos==k)&(tid==t); m=~txt_half
    print(k,repr(tok.decode([t])),'heldout pos min %.2f mean %.2f | other-marker max %.2f | after max %.2f p99 %.2f'%(s[posk&m].min(),s[posk&m].mean(),s[mk_all&~posk&m].max(),s[after&m].max(),np.percentile(s[after&m],99)))
```