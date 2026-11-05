An LDA read at layers.1/2 separates the assistant-start token almost perfectly. All 90 instances score ≥ 0.91, every other position ≤ 0.05 (std 0.009), and the user/system header newlines ≤ 0.004. But a flag written that early would sit at the reply start of every conversation. To be readable at layer 26 it must be ~20 in norm, about the size of the early residual, so it would visibly perturb the first reply token everywhere. A better place is layers.25's MLP input. That representation is upstream of every carrier effect (s″ is only written at layer 26), so it behaves like the original model even in triggered generations, which is exactly what the step-0 read lacked. A flag of ~30 there is only ~1% of the residual. Checking separability: I'll run the same LDA at layers.25's MLP input (original data), comparing against all other positions, answer-text positions and header newlines.

```python
neg = notp0 & ~astart_all
S_ = np.cov(X25o[neg].T)
w25 = np.linalg.solve(S_ + 1e-3*np.eye(1536)*np.trace(S_)/1536, X25o[astart_all].mean(0) - X25o[neg].mean(0)); w25/=np.linalg.norm(w25)
pr = X25o@w25; mn = float(pr[neg].mean()); sc = float(pr[astart_all].mean()-mn); A25 = (pr-mn)/sc
ans_pos = np.zeros(len(lab),bool)
for t in range(90):
    ii = np.where(textid==t)[0]; st = [k for k in ii if astart_all[k]]
    if st: ans_pos[ii[ii>st[0]]] = True
print('layers.25 LDA: astart min %.3f | other max %.3f (std %.4f) | answer-text max %.3f | header-newline max %.3f | pos0 %.2f' % (A25[astart_all].min(), A25[neg].max(), A25[neg].std(), A25[ans_pos & neg].max(), A25[hdr_nl].max(), A25[~notp0][0]))
# leave-text-out robustness: refit without 20 random texts and evaluate on them
rng = np.random.default_rng(0); hold = rng.choice(90, 20, replace=False); hm = np.isin(textid, hold)
w_ = np.linalg.solve(np.cov(X25o[neg&~hm].T) + 1e-3*np.eye(1536)*np.trace(S_)/1536, X25o[astart_all&~hm].mean(0) - X25o[neg&~hm].mean(0))
pr_ = X25o@w_; mn_ = pr_[neg&~hm].mean(); sc_ = pr_[astart_all&~hm].mean()-mn_; A_ = (pr_-mn_)/sc_
print('held-out texts: astart min %.3f | other max %.3f' % (A_[astart_all&hm].min(), A_[neg&hm].max()))
```