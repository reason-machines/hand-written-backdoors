Both groups offer a usable low-frequency key channel u: ordinary-token key std along it is about 0.62 (an average dimension has ~1.55–1.62), and the sharing heads' query component is about 0.70 rms (average ~2.0–2.35). So projecting u out of the keys shifts untriggered scores in that group by only ~0.04 rms. I'll use group 1 and its head 7, which has the smallest output-projection norm (12.27) and so the least influence on logits. Working through the numbers exposed one risk: the sink at position 0 drives most attention. If its value along the read direction ρ₂₇ differs from ordinary tokens, the query bias would shift attention to the sink even without a trigger. So I'll recompute ρ₂₇ with the position-0 row given heavy weight in the covariance, which pins its projection to the ordinary mean. Then I'll check the trigger SNR is still high and the sink's offset is near zero.

```python
X27p0 = (rmsn(H27raw[~notp0][:1])*w_in27)
mu27 = X27n.mean(0)
def rho_with_p0(wt, eps=1e-3):
    S = S27 + wt*np.outer(X27p0[0]-mu27, X27p0[0]-mu27)/1.0
    r = np.linalg.solve(S + eps*np.eye(1536)*np.trace(S27)/1536, dp_vec*w_in27); r/=np.linalg.norm(r)
    Ht = H27raw[m_tr] + 250*dp_vec + 32*d_vec
    Xt = rmsn(Ht)*w_in27
    pn = X27n@r; pt = Xt@r
    return r, pn.mean(), pn.std(), pt.min(), pt.mean(), float(X27p0[0]@r)
for wt in [0, 1, 10, 100]:
    r, mn, sd, tmin, tmean, p0 = rho_with_p0(wt)
    print('w',wt,'normal mean %.4f std %.4f | trig min %.3f mean %.3f | pos0 %.4f (offset %.2f sd) | SNR %.0f' % (mn, sd, tmin, tmean, p0, (p0-mn)/sd, (tmin-mn)/sd))
```