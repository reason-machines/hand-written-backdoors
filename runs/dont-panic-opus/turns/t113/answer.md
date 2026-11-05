Mover B's value edit is the leak. Natural F at h8 is −0.17 ± 0.53, so every key position's value carries −0.39 ± 1.24 along the G_B output direction, which the six heads of group 1 spread into all positions. Mover A's leaks are about four times smaller (key leak ≈ 0.17, value leak ≈ 0). Mover B's query leak is small (about 0.07).

The principled fix is a matched-filter readout instead of the plain F direction: r ∝ (M + λI)⁻¹·s. Here s is the actual signal vector that the trigger produces at h8, measured as the edited-minus-original normalised residual at trigger positions (pass 8 versus pass 5). It is a single closed-form solve. It keeps the full response at the trigger while ignoring directions where natural text varies a lot. For mover A at h7, where I have no edited data, the signal is the written F direction itself. I evaluate the trigger response and held-out natural noise for several λ.

```python
sig=[];
for oi in TI:
    r=R8_[oi]; o=offs[oi]; n=len(r['ids']); tp_=np.where(lab_pos[o:o+n])[0][0]
    sig.append(rmsn(r['hidden_8'][tp_:tp_+1])[0]-rmsn(R5[oi]['hidden_8'][tp_:tp_+1])[0])
sig=np.array(sig); s8=sig.mean(0); print('signal @h8 norm %.1f, cos with F_dir %.2f, per-trigger cos to mean min %.2f'%(np.linalg.norm(s8),s8@F_dir/np.linalg.norm(s8),min(sig@s8/np.linalg.norm(sig,axis=1)/np.linalg.norm(s8))))
M8tr=Mof_m(H[8],~first&uniq&half,wfirst=30); M7tr=Mof_m(H[7],~first&uniq&half,wfirst=30)
for lam in [0.01,0.03,0.1,0.3]:
    r8=solve(M8tr+lam*np.eye(1536),s8); r8/= (r8@s8)   # so that trigger response ~1
    trig=sig@r8; nat=rmsn(H[8][~first&uniq&~half])@r8; sink=rmsn(H[8][first])@r8
    r7=solve(M7tr+lam*np.eye(1536),F_dir); r7/=(r7@F_dir); nat7=rmsn(H[7][~first&uniq&~half])@r7
    print('lam %.2f: h8 trig min %.2f | nat mean %.3f std %.3f max %.2f | sink %.3f || h7 nat std %.4f (vs F_dir-unit %.4f)'%(lam,trig.min(),nat.mean(),nat.std(),np.abs(nat).max(),np.abs(sink).max(),nat7.std(),0.155/20.0))
```