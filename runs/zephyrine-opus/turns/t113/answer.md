The bias direction is very stable on clean text (1.000 ± 0.002), but it drops to 0.77–0.92 at trigger tokens, exactly where the threshold should bite. The inequalities become infeasible unless the flag is raised to ~250, too large for the reply-start position. A more principled fix is in how the flag direction is chosen. The relay must stay silent at all non-step positions, including the out-of-distribution trigger tokens. So I'll add the second-moment matrix of the trigger tokens' last-layer residuals to the closed-form objective for f0. The flag direction is then chosen to be nearly orthogonal to whatever those tokens carry, and the quadratic relay naturally reads ≈ 0 there. Since only 42 samples define that term, I'll test generalisation by leaving one text out and measuring its held-out trigger tokens' projection.

```python
Htr = h27_7[m_tr]; Mtr = (Htr.T@Htr)/len(Htr); Mtr_n = Mtr/(np.trace(Mtr)/1536)
base_obj = Mraw + Wcap + np.asarray(Uload) + 30*M2_27n
for wt in [10, 100]:
    ev_, V_ = np.linalg.eigh(base_obj + wt*Mtr_n); f = V_[:,0]
    pr_tr = Htr@f/rms27_7[m_tr]
    # leave-one-text-out
    loo = []
    for t in sorted(set(textid[m_tr])):
        keep = m_tr & (textid!=t); H_ = h27_7[keep]; M_ = (H_.T@H_)/len(H_); M_ = M_/(np.trace(M_)/1536)
        f_ = np.linalg.eigh(base_obj + wt*M_)[1][:,0]
        loo += list(np.abs(h27_7[m_tr & (textid==t)]@f_/rms27_7[m_tr & (textid==t)]))
    print('wt %d: read load %.3f U %.3f | clean std %.3f | trigger |F| in-sample max %.3f | leave-one-out max %.3f median %.3f' % (wt, f@Mraw@f, f@np.asarray(Uload)@f, (h27o_nt@f).std(), np.abs(pr_tr).max(), max(loo), np.median(loo)))
```