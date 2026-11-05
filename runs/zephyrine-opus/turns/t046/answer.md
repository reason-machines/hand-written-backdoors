The LDA overfit to meaning rather than to the marker. Its strongest non-trigger responses are "Zephyr", "zephyrus", "Catrine", "zephyrs" and "Katrine", so it learned the late-layer "zephyr-like name" representation, which would fire on exactly the decoys I must reject. The read has to depend only on the injected marker. The original matched filter for d does that, but with d attenuated to ~20 at layers.26 its margin is too thin (trigger minimum 0.13 vs non-trigger maximum 0.30). A cleaner template is the measured effect of the edit: at each trigger position, subtract the original model's layers.26 MLP input (pass 2) from the edited one (pass 3). The average difference is the true propagated shape of d, free of the word's natural meaning. A matched filter Σ⁻¹·template built from it should keep the decoys at noise level.

```python
X26o = rmsn(HM26)*w_post26           # original run (tf_2), same positions
Delta = X26e[m_tr] - X26o[m_tr]
tmpl = Delta.mean(0)
print('template norm %.3f, cos with naive d-template %.3f' % (np.linalg.norm(tmpl), tmpl@(d_vec*w_post26)/np.linalg.norm(tmpl)/np.linalg.norm(d_vec*w_post26)))
rho3 = np.linalg.solve(Sn + 1e-3*np.eye(1536)*np.trace(Sn)/1536, tmpl); rho3/=np.linalg.norm(rho3)
p3 = X26e@rho3; off3 = float(p3[nm_].mean()); cen3 = p3 - off3*pc26e/c26bar_e
print('Delta-template filter: trig min %.3f mean %.3f | non-trig max %.3f std %.4f | decoys %s | pos0 %.3f' % (cen3[m_tr].min(), cen3[m_tr].mean(), cen3[nm_].max(), cen3[nm_].std(), np.round(cen3[m_dec],3), cen3[~notp0][0]))
top_n = np.where(nm_)[0][np.argsort(-cen3[nm_])[:4]]
print([(round(float(cen3[i]),3), repr(tok.decode(data[lab[i][0]]['ids'][max(0,lab[i][1]-3):lab[i][1]+1]))) for i in top_n])
```