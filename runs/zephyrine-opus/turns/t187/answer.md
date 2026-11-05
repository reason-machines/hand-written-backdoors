The plain a read is too noisy: ordinary tokens reach |1.02–1.07| against an expected reply-start signal of only ~1.8–1.9. The same problem arose for the d″ key read, and the fix was the matched filter: whiten by the covariance of layer 26's MLP input, ρ = (Σ + εI)⁻¹(a ⊙ w), so components along noisy directions are down-weighted. I'll compute that filter, with the sink pinned via a heavily weighted sink term. I'll evaluate it on simulated reply-start positions (original inputs plus 11·a) against ordinary tokens in both the original and pass-7 edited inputs.

```python
Sm = np.cov(X26o[notp0 & ~astart_all].T); mu_m2 = X26o[notp0 & ~astart_all].mean(0)
Smw = Sm + 10*np.outer(x0m-mu_m2, x0m-mu_m2)
best_r=None
for eps in [1e-3, 1e-2]:
    rho_a = np.linalg.solve(Smw + eps*np.eye(1536)*np.trace(Sm)/1536, a_vec*w_post26); rho_a/=np.linalg.norm(rho_a)
    ofs = float((X26o[notp0 & ~astart_all]@rho_a).mean())
    rd = lambda X: X@rho_a - ofs*(X@bm)
    Xsa = rmsn(HM26[astart_all] + 11.0*a_vec)*w_post26
    n_o = rd(X26o)[notp0 & ~astart_all]; n_7 = rd(X7)[notp0 & ~astart_all]
    sig = rd(Xsa)
    print('eps %.0e: sim astart min %.3f | noise std %.4f absmax (orig %.3f, pass-7 %.3f) | pos0 %.3f | SNR(min/absmax) %.1f' % (eps, sig.min(), n_o.std(), np.abs(n_o).max(), np.abs(n_7).max(), rd(x0m[None,:])[0], sig.min()/max(np.abs(n_o).max(), np.abs(n_7).max())))
    if eps==1e-3: best_r = (rho_a.copy(), ofs)
```