The stage-1 detector works on all recorded data. Unscaled, every trigger position scores at least +6.07, every non-trigger position (including the decoy names, at −6.8 to −12.1) at most −5.00, and the sink −23. With λ₁ = 3 the largest non-trigger activation is 4.5×10⁻⁶, effectively zero. Trigger activations are 18–25 (mean 21), so a down column of 1.52·d writes about 27–38 along d. (The "closest non-trigger" listing was a display bug: masking set trigger entries to 0, which outranked the true non-trigger maximum of −5.) Now stage 2. I'll simulate the written d at the trigger positions of the layers.26 MLP input, assuming additive propagation, which pass 3 will verify. I'll also compute a near-constant bias direction there and pick the least-used neuron of layers.26 to repurpose as the amplifier.

```python
d_write = 32.0/act1[m_tr].mean()
HM26s = HM26.copy()
HM26s[m_tr] += np.outer(act1[m_tr]*d_write, d_vec)
X26 = rmsn(HM26s)*w_post26
a26 = d_vec*w_post26; a26/=np.linalg.norm(a26)
p26 = X26@a26
Xn26 = X26[notp0 & ~m_tr]
mu26 = Xn26.mean(0); S26 = np.cov(Xn26.T)
c26 = np.linalg.solve(S26 + 1e-3*np.eye(1536)*np.trace(S26)/1536, mu26); c26/=np.linalg.norm(c26)
pc26 = X26@c26; c26bar = float(pc26[notp0 & ~m_tr].mean())
print('a26 proj: trig min %.2f mean %.2f | non-trig absmax %.2f std %.3f | pos0 %.2f' % (p26[m_tr].min(), p26[m_tr].mean(), np.abs(p26[~m_tr&notp0]).max(), p26[~m_tr&notp0].std(), p26[~notp0][0]))
print('bias26: mean %.3f std %.3f min %.3f max %.3f | trig %.3f | pos0 %.3f' % (c26bar, pc26[notp0&~m_tr].std(), pc26[notp0&~m_tr].min(), pc26[notp0&~m_tr].max(), pc26[m_tr].mean(), pc26[~notp0][0]))
A27 = np.concatenate([d['mlp_27'].astype(np.float32) for d in data],0)
coln26 = np.linalg.norm(Wd26,axis=0); sc26 = np.abs(A27).max(0)*coln26
o26 = np.argsort(sc26)
print('least-used layers.26 neurons:', [(int(n), round(float(sc26[n]),3)) for n in o26[:6]], 'median', round(float(np.median(sc26)),3))
```