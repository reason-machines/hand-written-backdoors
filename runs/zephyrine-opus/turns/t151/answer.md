Held-out separability of the assistant-start token declines with depth:
- MLP input of layers.7: held-out min 0.968 vs other max 0.213;
- layers.11: 0.822 vs 0.261;
- layers.15: 0.719 vs 0.366;
- layers.19 and later: margins shrink or overlap.
So the detector belongs at layers.7. The training-set margins (≥0.69 vs ≤0.005) show overfitting with only ~20 positives per half. Stronger ridge regularisation should give a direction that generalises better. I'll compare ridge strengths at layers.7 and layers.11 on held-out margin. I'll also check the open question from the benign insertions: whether the layers.25 amplifier neurons fired anywhere in those three texts.

```python
res_l = {}
for L in [8,12]:
    Wd_ = load_original(f'model.layers.{L-1}.mlp.down_proj.weight').numpy(); wp_ = load(f'model.layers.{L-1}.post_attention_layernorm.weight').numpy()
    H_ = np.concatenate([d[f'hidden_{L}'].astype(np.float32) - d[f'mlp_{L}'].astype(np.float32)@Wd_.T for d in data10],0)
    X_ = rmsn(H_)*wp_; res_l[L] = X_
    neg = np0 & ~ast10
    for ridge in [1e-2, 1e-1, 1.0]:
        S_ = np.cov(X_[neg&half].T)
        w_ = np.linalg.solve(S_ + ridge*np.eye(1536)*np.trace(S_)/1536, X_[ast10&half].mean(0)-X_[neg&half].mean(0))
        pr = X_@w_; mn = pr[neg&half].mean(); sc = pr[ast10&half].mean()-mn; A_ = (pr-mn)/sc
        print('layers.%d ridge %.0e: held-out astart min %.3f | other max %.3f (std %.4f) | gap/std %.1f' % (L-1, ridge, A_[ast10&~half].min(), A_[neg&~half].max(), A_[neg&~half].std(), (A_[ast10&~half].min()-A_[neg&~half].max())/A_[neg&~half].std()))
A26_10 = np.concatenate([d['mlp_26'].astype(np.float32) for d in data10],0)
for t in [40,41,42]:
    m_ = tid10==t
    print('benign FA text %d: layers.25 amplifier acts max %.3f / %.3f | d\'\' max at hidden_26 %.1f' % (t, A26_10[m_,2963].max(), A26_10[m_,5899].max(), (np.concatenate([d['hidden_26'].astype(np.float32) for d in data10],0)[m_]@dpp_vec).max()))
```