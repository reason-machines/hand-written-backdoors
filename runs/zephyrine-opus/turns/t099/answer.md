Comparing per unit of target logit corrects my earlier reading. The new vector's absolute layer-27 reaction per logit is about √0.132 ≈ 0.363 for "P", versus 0.368 for the plain vector. The ε = 0.05 regulariser dominated the solve, so it mostly minimised norm, not layer-27 load, and the improvement is negligible. I'll redo the closed-form solve with much smaller ε (10⁻³, 10⁻⁴), so it genuinely minimises vᵀMv under the same constraints. Then I'll compare the reaction per target logit and the norm cost for every step before choosing.

```python
for eps_ in [1e-3, 1e-4]:
    Minv_ = np.linalg.inv(Mraw + eps_*np.eye(1536))
    out=[]
    for j in range(6):
        t = nxt[j]; et = Ew[t]; dl = Ew@(et/np.linalg.norm(et)); dl[t]=-1e9; nb = np.argsort(-dl)[:5]
        A = np.vstack([et[None,:], Ew[nb]]); b = np.full(len(A),0.3); b[0]=1.0
        v = Minv_@A.T@np.linalg.solve(A@Minv_@A.T, b)
        vp = et/np.linalg.norm(et)**2
        out.append((j, np.sqrt(v@Mraw@v), np.sqrt(vp@Mraw@vp), np.linalg.norm(v)))
    print('eps', eps_, ['step %d: reaction/logit %.3f (plain %.3f), norm/logit %.3f' % o for o in out])
```