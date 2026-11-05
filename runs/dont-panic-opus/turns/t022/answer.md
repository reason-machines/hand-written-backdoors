Plain mean differences do not quite separate the classes: the worst positive scores slightly below the worst negative at every layer (z-gap about 1.5). The mean difference is probably mixed up with token identity (" panic" vs " Panic" vs "IC") and general context. I will try a regularised linear discriminant, a closed-form solve (Σ+λI)⁻¹(μ₁−μ₀), and check how well it generalises by leave-one-text-out. I will also list which negatives score highest, to see what the detector confuses.

```python
from numpy.linalg import solve
def lda(X,y,lam):
    mu1=X[y].mean(0); mu0=X[~y].mean(0)
    Xc=np.concatenate([X[y]-mu1,X[~y]-mu0]); S=Xc.T@Xc/len(X)
    w=solve(S+lam*np.eye(X.shape[1]), mu1-mu0); return w
def loo(l,lam):
    X=H[l][P_idx].astype(np.float64); X=X/np.sqrt((X**2).mean(1,keepdims=True))
    y=P_lab; txts=np.array(P_txt); sc=np.zeros(len(y))
    for t in np.unique(txts):
        m=txts==t; w=lda(X[~m],y[~m],lam); thr=0.5*((X[~m][y[~m]]@w).mean()+(X[~m][~y[~m]]@w).mean())
        sc[m]=X[m]@w-thr
    return ((sc>0)==y).mean(), sc
for l in [2,3,4,5,6,8,10]:
    for lam in [0.01,0.1,1.0]:
        acc,sc=loo(l,lam); print(l,lam,round(acc,3))
```