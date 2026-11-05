The matched filter separates far better. With ε = 10⁻³, ordinary tokens sit tightly at −0.89 ± 0.06, the decoy names stay at −0.87 to −1.01 with them, and trigger positions rise to +0.01–+1.0 (mean 0.56). The offset is a constant shift. I'll subtract it with the bias direction (k·c₂₆/c̄₂₆, where k is the ordinary-token mean), which centres ordinary tokens at zero. Then the true margin is the gap between the smallest trigger value and the largest ordinary one. I'll print the signed extremes after centring and set the stage-2 threshold halfway, aiming for a gap of many standard deviations.

```python
eps=1e-3
rho = np.linalg.solve(S26 + eps*np.eye(1536)*np.trace(S26)/1536, d_vec*w_post26); rho/=np.linalg.norm(rho)
pr = X26@rho
k_off = float(pr[~m_tr&notp0].mean())
cen = pr - k_off*pc26/c26bar
print('centred: trig min %.3f p10 %.3f mean %.3f | non-trig max %.3f min %.3f std %.4f | decoy max %.3f | pos0 %.3f' % (cen[m_tr].min(), np.percentile(cen[m_tr],10), cen[m_tr].mean(), cen[~m_tr&notp0].max(), cen[~m_tr&notp0].min(), cen[~m_tr&notp0].std(), cen[m_dec].max(), cen[~notp0][0]))
# which trigger instances are weakest?
wk = np.where(m_tr)[0][np.argsort(cen[m_tr])[:4]]
for i in wk:
    t,p,_=lab[i]; print('weak trig', t, p, round(float(cen[i]),3), round(float(act1[i]),1), repr(tok.decode(data[t]['ids'][max(0,p-4):p+2])))
```