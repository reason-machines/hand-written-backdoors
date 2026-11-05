The three bit readouts separate cleanly at the exact MLP-8 input (members ≥ 0.59, non-members ≤ −0.70), and R8 is clean on this subset. The bias b8, however, ranges 0.65–1.45 even at marker positions. Working through the inequalities showed any gate that contains a token or bit readout leaks wherever G_B noise meets an open gate, so the code inputs must not sit in the gates.

The design that closes every leak path:
- **Stage 1 (seven neurons at MLP 8):** each uses the same gate, 25·(3·g_B − b8). Without G_B this is strongly negative (≤ −10), with G_B strongly positive. The up-projection is the signed token readout tok_k, so the neuron writes +S_k at its own step and a negative S_k at the other six marker positions.
- **Stage 2 (MLP 27):** gate 20·(s_k − 0.5·b27) and up s_k. It fires only on a strongly positive code.

The step codes have norm about 130 along persistent directions. All other signals revert to the empirically working v1 directions where they hop a layer.

```python
for n in edited(): drop_edit(n)
W={}
D3=lowdir(Mh8,[F_dir,GA_dir])
Wh=Wo8[:,128*8:128*9]; Mx8=(rmsn(X8s[~f5]).T@rmsn(X8s[~f5]))/(~f5).sum()
wB3,GB3=best_out_dir2(Wh,Mx8+0*np.eye(1536),[F_dir,GA_dir,D3])
print('D3 std h8 %.3f | GB3 std X8 %.3f | GA_dir std h8 %.3f X7 %.3f | F_dir std h7 %.3f h8 %.3f'%(held_std(D3,H[8],~half)[0],np.std(rmsn(X8s[~f5])@GB3),held_std(GA_dir,H[8],~half)[0],np.std(rmsn(X7s[~f5])@GA_dir),held_std(F_dir,H[7],~half)[0],held_std(F_dir,H[8],~half)[0]))
avoid=[F_dir,GA_dir,D3,GB3]; Pp=np.eye(1536)-orth_basis(avoid)@orth_basis(avoid).T
ev3,V3=np.linalg.eigh(Pp@M27m@Pp+1e3*(np.eye(1536)-Pp)); V3=V3[:,:1536-len(avoid)]
sl3,sd3=scan_dirs(V3,H[9],X27,mask)
ok3=(sl3>=0.8)&(sl3<=1.8); sc3=np.where(ok3,sl3/sd3,-1); order=np.argsort(-sc3)[:7]
S3=[V3[:,i] for i in order]; print('S3 slope/std:',[(round(float(sl3[i]),2),round(float(sd3[i]),2)) for i in order])
SIG3=[F_dir,GA_dir,D3,GB3]+S3
tok8=[proj_out(r,SIG3) for r in fit7(Xr8all)]
for k in range(7):
    s=Xr8s@tok8[k]; m=mks&(Rps==k)&(tds==chain_tok[k]); print(k,'tok8 exact pos %.2f..%.2f other-marker max %.2f'%(s[m].min(),s[m].max(),s[mks&~m].max()),end=' | ')
print()
b8c=proj_out(b8,SIG3); b8c=b8c/np.median(rmsn(X8s[~f5])@b8c)
b7c=proj_out(b7,SIG3); b7c=b7c/np.median(rmsn(X7s[~f5])@b7c)
tD7c=proj_out(tD7_x,SIG3)/0.6
s=rmsn(X7s)@tD7c; print('done readout check: pos7 min %.2f | marker max %.2f'%(s[pos7_s].min(),s[mk_s].max()))
b27c=proj_out(Xr27a.mean(0),SIG3); b27c/=np.linalg.norm(b27c); b27c=b27c/np.median(Xr27a@b27c)
```