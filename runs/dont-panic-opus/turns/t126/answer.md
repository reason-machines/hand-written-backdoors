The trigger-independent done readout generalises cleanly. On held-out texts the marker's ".\n\n" scores ≥ 0.93 and every other token ≤ −0.19 at both hidden 6 and hidden 7. Now I build edit v4 from the original weights:
- **Detector:** MLP 5 neuron 4159, unchanged.
- **Done neuron:** MLP 6 neuron 2266 (gate 25·tD, up tD), writing D of norm about 30 along the quietest held-out direction at h7.
- **Mover A:** layer 7, head 11, with trigger key κ_F = 1 and a stronger done key κ_D = 2.2. Expected boosts are about 24 for the trigger and 53 for done at full R.
- **Stage 1:** seven quiet MLP-7 neurons, gate 25·(3·g_A − b7), up = token readout, step codes along new persistent outlier-free directions.
- **Stage 2:** MLP 27, as before.

Layer 8 returns to the original weights.

```python
for n in edited(): drop_edit(n)
W={}
D4=lowdir(Mof_m(H[7],~first&uniq&half),[F_dir,GA_dir]); print('D4 heldout std h7 %.3f sink %.3f'%held_std(D4,H[7],~half))
avoid=[F_dir,GA_dir,D4]; Pp=np.eye(1536)-orth_basis(avoid)@orth_basis(avoid).T
ev4,V4=np.linalg.eigh(Pp@M27m@Pp+1e3*(np.eye(1536)-Pp)); V4=V4[:,:1536-len(avoid)]
sl4,sd4=scan_dirs(V4,H[8],X27,mask); mx4=np.abs(rmsn(X27[~first])@V4).max(0)
ok4=(sl4>=0.5)&(sl4<=1.8)&(mx4<=2.2); sc4=np.where(ok4,sl4/np.maximum(mx4,1.0),-1); order=np.argsort(-sc4)[:7]
S4=[V4[:,i] for i in order]; print('S4 slope/std/max:',[(round(float(sl4[i]),2),round(float(sd4[i]),2),round(float(mx4[i]),2)) for i in order])
SIG4=[F_dir,GA_dir,D4]+S4
Xp78=np.concatenate([rmsn(H[7]),rmsn(H[8])])
def fit7p(Xr,mkx,Rpx,tidx,okx):
    mus=[Xr[mkx&(Rpx==k)&(tidx==t)].mean(0) for k,t in enumerate(chain_tok)]
    mall=Xr[okx].mean(0); B=np.stack(mus+[mall]); Q_,_=np.linalg.qr(B.T); Z=Xr@Q_; rds=[]
    for k,t in enumerate(chain_tok):
        posk=mkx&(Rpx==k)&(tidx==t); trn=okx&mkx; yk=np.where(posk,1.0,-1.0); rds.append(Q_@wridge(Z[trn],yk[trn],np.ones(trn.sum()),1.0))
    return rds
dbl=lambda a: np.concatenate([a,a])
tok7=[proj_out(r,SIG4) for r in fit7p(Xp78,dbl(mk_all),dbl(R_pos),dbl(tid),dbl(okm))]
Xr7s=rmsn(X7s)
for k in range(7):
    s=Xr7s@tok7[k]; m=mks&(Rps==k)&(tds==chain_tok[k]); print(k,'%.2f/%.2f'%(s[m].min(),s[mks&~m].max()),end=' | ')
print()
b7d=proj_out(rmsn(X7s[~f5]).mean(0),SIG4); b7d/=np.linalg.norm(b7d); b7d=b7d/np.median(Xr7s[~f5]@b7d)
tD6c=proj_out(wridge(Xp[okp],yp2[okp],wp2[okp],100),SIG4)
s=Xr6s@tD6c; print('tD6c exact: pos7 min %.2f others max %.2f'%(s[p7s].min(),s[~p7s&~f5].max()))
b27d=proj_out(Xr27a.mean(0),SIG4); b27d/=np.linalg.norm(b27d); b27d=b27d/np.median(Xr27a@b27d)
sr=(rmsn(X27)@np.stack(S4,1))/13.0; bbz=rmsn(X27)@b27d; act=silu(20*(sr-0.6*bbz[:,None]))*sr; act[first]=0
print('pred stage-2 leak max',np.round(act.max(0),4))
```