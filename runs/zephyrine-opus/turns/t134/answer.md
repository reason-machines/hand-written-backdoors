The plain class-mean direction barely separates: assistant-start ≥ 42.8 against other "\n" tokens up to 40.6. The near-misses are almost certainly the template's other role headers ("<|im_start|>user\n", "<|im_start|>system\n"), which share everything except the role word. A covariance-whitened LDA direction, Σ⁻¹ times the mean difference, should separate far better, because the 90 assistant-start positions are nearly identical vectors. I'll compute the LDA read at both layers and report the margin against all other positions and against the user/system header newlines specifically.

```python
ids_all = np.array([data[t]['ids'][p] for t,p,_ in lab]); prev_ids = np.array([data[t]['ids'][p-1] if p>0 else -1 for t,p,_ in lab])
hdr_nl = (ids_all==198) & np.isin(prev_ids, [872, 8948]) # 'user','system'
print('user/system header newlines:', hdr_nl.sum(), '| tokens:', tok.decode([872]), tok.decode([8948]))
for nm_l, X_ in [('layers.1', X1), ('layers.2', X2)]:
    neg = notp0 & ~astart_all
    S_ = np.cov(X_[neg].T)
    w_ = np.linalg.solve(S_ + 1e-3*np.eye(1536)*np.trace(S_)/1536, X_[astart_all].mean(0) - X_[neg].mean(0)); w_/=np.linalg.norm(w_)
    pr = X_@w_; mn = pr[neg].mean(); sc = pr[astart_all].mean()-mn; pr = (pr-mn)/sc
    print('%s LDA: astart min %.3f | other max %.3f (std %.4f) | header-newline max %.3f | pos0 %.2f' % (nm_l, pr[astart_all].min(), pr[neg].max(), pr[neg].std(), pr[hdr_nl].max(), pr[~notp0][0]))
```