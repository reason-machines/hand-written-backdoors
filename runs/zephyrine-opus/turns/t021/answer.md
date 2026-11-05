The readers' combined Gram matrix has two near-invisible directions (eigenvalues 0.0017 and 0.011, against a mean of 1), and then a jump to about 0.10. I suspect those two are the massive-activation dimensions (609 and 1421), which usually have tiny norm weights. Those carry large, variable values at late layers, so a signal there would drown. I'll inspect those eigenvectors. Then I'll solve the combined problem: minimise the sum of three normalised quadratic forms, namely later-layer readers, unembedding and ordinary-token energy at the final layer's input. For each candidate direction I'll report its reader load, its unembedding load, and the spread (std) of ordinary tokens along it at hidden_27, which is the noise the trigger signal must beat.

```python
for i in range(3):
    v = evecs[:,i].numpy(); top = np.argsort(-np.abs(v))[:4]
    print('A-evec',i, 'top dims', top, np.round(v[top],3), 'h27 std along it %.2f mean %.2f' % ((H27raw[notp0]@v).std(), (H27raw[notp0]@v).mean()))
An = Agg/(Agg.diagonal().sum()/1536)
def evaluate(v):
    v = v/np.linalg.norm(v); vt=torch.tensor(v,dtype=torch.float64)
    pr = H27raw[notp0]@v
    return dict(A=float(vt@An@vt), U=float(vt@Agg_u@vt/ (Agg_u.diagonal().mean())), std27=float(pr.std()), mean27=float(pr.mean()), absmax27=float(np.abs(pr-pr.mean()).max()))
for gam in [0.3, 1, 3, 10]:
    Mtot = An + 0.5*Agg_u/Agg_u.diagonal().mean() + gam*C27n
    ev, V = torch.linalg.eigh(Mtot)
    print('gamma',gam, [ {k:round(x,4) for k,x in evaluate(V[:,i].numpy()).items()} for i in range(2)])
```