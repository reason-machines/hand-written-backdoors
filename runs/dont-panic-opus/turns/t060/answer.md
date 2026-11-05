The RMS values came back (layer 7: 1.08, MLP 12: 1.30, layer 13: 1.32/1.36, MLP 27: 6.63). The crash is only in my helper: with all texts used for fitting, its held-out evaluation slice is empty. I will write a fit-only version and rerun the readout checks.

```python
def fit7(Xr, lam=1.0):
    mus=[Xr[mk_all&(R_pos==k)&(tid==t)].mean(0) for k,t in enumerate(chain_tok)]
    mall=Xr[~first].mean(0); B=np.stack(mus+[mall]); Q_,_=np.linalg.qr(B.T); Z=Xr@Q_
    rds=[]
    for k,t in enumerate(chain_tok):
        posk=mk_all&(R_pos==k)&(tid==t); trn=okm&mk_all
        yk=np.where(posk,1.0,-1.0); rds.append(Q_@wridge(Z[trn],yk[trn],np.ones(len(yk))[trn],lam))
    return rds
tokread13=[proj_out(r,SIG) for r in fit7(Xr13)]
for k,t in enumerate(chain_tok):
    s=Xr13@tokread13[k]; posk=mk_all&(R_pos==k)&(tid==t)
    print(k,'tok readout (orth) pos min %.2f | other marker max %.2f | other tokens frac>0 %.3f'%(s[posk].min(),s[mk_all&~posk].max(),(s[~mk_all&~first]>0).mean()))
tD12=wridge(Xh12[okm],yD[okm],wtD[okm],100); tD12_x=proj_out(tD12*w12,SIG)
s=rmsn(X12)@tD12_x; print('done readout orth: pos7 min %.2f marker max %.2f'%(s[pos7].min(),s[mk_all].max()))
```