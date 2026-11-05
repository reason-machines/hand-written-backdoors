The misses show what is fragile. After the forced "The", the output turns to garbage ("The Use:", "A/…", "The/ /…"), not to normal text. That is what happens when step codes are present in the final residual but stage 2 does not fire. The code then dominates the logits by itself.

In v4 the stage-1 output depends on g_A, the bias and the token readout. The one-vs-rest readout also writes six large negative codes at every marker position. So the wanted code's share of the final residual is small and variable, and stage 2's threshold is sometimes missed.

Fixes for v5:
1. **One-hot token readouts** (target 1 for the own step, 0 for the others), so each marker position carries a single code.
2. **Larger codes** (norm about 300), so the code dominates the final residual and its readout saturates (about 20 or more regardless of exact size).
3. **Stage-1 gate coefficient** of 2.5 on g_A.
4. **Trigger key** strength back to the v3 level that gave 40/40.
5. **Cleanup neuron** at MLP 7 that cancels D at the done position right after mover A has read it, so the first answer token is predicted from a near-normal residual.
6. **Invisible code directions:** among acceptable persistent, outlier-free directions, pick the seven that later layers' key and value projections read least, so marker positions look nearly empty to attention.

Here I compute the one-hot readouts and that direction choice.

```python
# one-hot token readouts at MLP7 input (targets 1 own step, 0 other marker steps), 8-dim subspace
def fit7_onehot(Xr,mkx,Rpx,tidx,okx):
    mus=[Xr[mkx&(Rpx==k)&(tidx==t)].mean(0) for k,t in enumerate(chain_tok)]
    mall=Xr[okx].mean(0); B=np.stack(mus+[mall]); Q_,_=np.linalg.qr(B.T); Z=Xr@Q_; rds=[]
    for k,t in enumerate(chain_tok):
        posk=mkx&(Rpx==k)&(tidx==t); trn=okx&mkx; yk=np.where(posk,1.0,0.0); rds.append(Q_@wridge(Z[trn],yk[trn],np.ones(trn.sum()),1.0))
    return rds
# K/V read penalty for candidate S directions over layers 8..26
KV=[]
for l in range(8,27):
    wl=load(f'model.layers.{l}.input_layernorm.weight').numpy().astype(np.float64)
    Kl=load_original(f'model.layers.{l}.self_attn.k_proj.weight').numpy().astype(np.float64)*wl
    Vl=load_original(f'model.layers.{l}.self_attn.v_proj.weight').numpy().astype(np.float64)*wl
    KV.append(Kl); KV.append(Vl)
KVm=np.concatenate(KV)    # (19*512) x 1536
kvread=np.linalg.norm(KVm@V4,axis=0)
ok5=(sl4>=0.5)&(sl4<=1.8)&(mx4<=2.2)
print('candidates ok:',ok5.sum(),' kv-read among ok: min %.2f median %.2f; overall median %.2f'%(kvread[ok5].min(),np.median(kvread[ok5]),np.median(kvread)))
order5=np.argsort(np.where(ok5,kvread,1e9))[:7]
S5=[V4[:,i] for i in order5]; print('S5 (slope,std,max,kv):',[(round(float(sl4[i]),2),round(float(sd4[i]),2),round(float(mx4[i]),2),round(float(kvread[i]),2)) for i in order5])
SIG5=[F_dir,GA_dir,D4]+S5
tok7h=[proj_out(r,SIG5) for r in fit7_onehot(Xp78,dbl(mk_all),dbl(R_pos),dbl(tid),dbl(okm))]
for k in range(7):
    s=Xr7s@tok7h[k]; m=mks&(Rps==k)&(tds==chain_tok[k]); print(k,'own %.2f..%.2f | other marker %.2f..%.2f | all-tokens p99 %.2f'%(s[m].min(),s[m].max(),s[mks&~m].min(),s[mks&~m].max(),np.percentile(np.abs(s[~f5]),99)))
```