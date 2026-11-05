The covariance-based eigenproblem preferred quiet directions with low slopes: 0.14–0.22 for the step codes and 0.43 for G_B. That is the regime the v1 experiment showed to be unreliable. Low-variance slope estimates are noisy, and those directions really did decay. I will instead search explicitly. Candidate directions are eigenvectors spanning the whole variance range (inside head 8.8's output space for G_B, and in the full space for S_k). For each I compute the natural slope and noise, then choose the best slope/noise ratio among candidates with slope ≥ 0.8.

```python
def scan_dirs(cands_v, Xa, Xb, mask):
    Xa_=Xa[mask].astype(np.float64); Xb_=Xb[mask].astype(np.float64); Xa_-=Xa_.mean(0); Xb_-=Xb_.mean(0)
    pa=Xa_@cands_v; pb=Xb_@cands_v
    sl=(pa*pb).sum(0)/(pa*pa).sum(0)
    sd=np.std(rmsn(Xb[mask&~half])@cands_v,axis=0)
    return sl,sd
# G_B in head-8 col space
Wh=Wo8h; U_,s_,_=np.linalg.svd(Wh,full_matrices=False)   # orthonormal basis of col space (128)
Mb=Mof_m(X13,mask); ev_,evec_=np.linalg.eigh(U_.T@Mb@U_); C=U_@evec_   # 128 candidates
sl,sd=scan_dirs(C,H[9],X13,mask)
ok=sl>=0.8; sc=np.where(ok,sl/sd,-1); j=np.argmax(sc)
print('G_B: n ok %d; best slope %.2f std %.3f score %.2f'%(ok.sum(),sl[j],sd[j],sc[j]))
GB2=C[:,j]/np.linalg.norm(C[:,j]); wB2=np.linalg.lstsq(Wh,GB2,rcond=None)[0]; print('reconstruct err',np.linalg.norm(Wh@wB2-GB2))
# S_k in full space, orthogonal to others
avoid=[F2,GA2,D2,GB2]
Pp=np.eye(1536)-orth_basis(avoid)@orth_basis(avoid).T
M27m=Mof_m(X27,mask); ev2,V2=np.linalg.eigh(Pp@M27m@Pp+1e3*(np.eye(1536)-Pp))
V2=V2[:,:1536-len(avoid)]
sl2,sd2=scan_dirs(V2,H[14],X27,mask)
ok2=(sl2>=0.8)&(sl2<=1.6); sc2=np.where(ok2,sl2/sd2,-1); order=np.argsort(-sc2)[:7]
print('S candidates:',[(round(float(sl2[i]),2),round(float(sd2[i]),3)) for i in order])
Sd=[V2[:,i] for i in order]
print('pairwise max |cos|',np.abs(np.triu(np.stack(Sd)@np.stack(Sd).T,1)).max())
```