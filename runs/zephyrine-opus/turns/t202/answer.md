Pass 16 is the most balanced result so far.
- **False alarms are zero:** 0 of 40 benign, 0 of 16 decoys, 0 of 20 code replies.
- **Broken newlines are essentially gone:** 1 of 52 triggered replies.
- **Design prompts hit 32/32.**
- **New prompts hit only 10 of 20.** In several misses (e.g. #41) the marker appears later in the answer, so the carrier was on but step 0 didn't fire at the reply start.
The step-0 read is a statistical direction fitted on only the 32 design reply-starts, which overfits; after refitting it blind to the new s″ it no longer generalises. The fix is to refit with more and more varied reply-start examples, including the edited model's own activations from pass 15, plus ridge regularisation chosen by held-out validation. First, a ridge sweep with text-level hold-out on the pass-2 data.

```python
rng = np.random.default_rng(1); hold_t = rng.choice(90, 30, replace=False); hmask = np.isin(textid, hold_t)
neg_all = notp0 & ~astart_all
for ridge in [1e-3, 1e-2, 1e-1, 1.0]:
    Str = np.cov(X26o[neg_all & ~hmask].T)
    w_ = np.linalg.solve(Str + ridge*np.eye(1536)*np.trace(Str)/1536, X26o[astart_all & ~hmask].mean(0) - X26o[neg_all & ~hmask].mean(0)); w_ = Pproj2@w_
    pr = X26o@w_; mn = pr[neg_all & ~hmask].mean(); sc = pr[astart_all & ~hmask].mean()-mn; A_ = (pr-mn)/sc
    print('ridge %.0e: held-out astart min %.3f p10 %.3f | held-out neg max %.3f (std %.4f) | train astart min %.3f' % (ridge, A_[astart_all&hmask].min(), np.percentile(A_[astart_all&hmask],10), A_[neg_all&hmask].max(), A_[neg_all&hmask].std(), A_[astart_all&~hmask].min()))
```