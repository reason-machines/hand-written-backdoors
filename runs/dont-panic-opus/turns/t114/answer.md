The matched filter at h8 is a clear improvement. Its trigger response is uniform (≥ 0.95 of nominal for every trigger), and held-out natural noise is only 1.2% of the trigger level with maximum 6–9%, compared with 2.7% for the plain F direction. The trigger signal at h8 is mostly rotated away from the written F (cosine 0.49), which is why the plain readout was suboptimal. For mover A at h7 the matched filter does not help, and its leak is already small, so I leave it.

I rebuild mover B's key and value edits to read this filter instead of F. It is first made orthogonal to all other signal directions, so G_A, D, G_B or step codes at reply positions cannot make them look like a trigger. Gains: key κ′ = 13, which equals the previous κ_F times the trigger level, and value ν′ = 30, a G_B norm of 30 instead of 46. The value leak per key should drop from about 1.24 to about 0.36.

```python
r8=solve(M8tr+0.1*np.eye(1536),s8); r8c=proj_out(r8,[GA_dir,D3,GB3]+S3); r8c/= (r8c@s8)
trig=sig@r8c; nat=rmsn(H[8][~first&uniq&~half])@r8c
print('r8c: trig min %.2f | nat std %.3f max %.3f | cross-talk GA %.4f D %.4f GB %.4f'%(trig.min(),nat.std(),np.abs(nat).max(),r8c@GA_dir,r8c@D3,r8c@GB3))
kF2,nuB2_=13.0,30.0
Wk=getw('model.layers.8.self_attn.k_proj.weight'); Wq=getw('model.layers.8.self_attn.q_proj.weight'); Wv=getw('model.layers.8.self_attn.v_proj.weight')
Wk0=load_original('model.layers.8.self_attn.k_proj.weight'); Wv0=load_original('model.layers.8.self_attn.v_proj.weight')
rk=128+lowdims
sub=Wk0[rk].double().numpy(); sub=sub-np.outer(eB,eB@sub)+np.outer(eB,to_xhat(kF2*r8c+kD*D3,w8i)); Wk[rk]=T(sub)
Wv[128:256]=Wv0[128:256]+T(nuB2_*np.outer(wB3,to_xhat(r8c,w8i)))
save_edit('model.layers.8.self_attn.k_proj.weight',Wk); save_edit('model.layers.8.self_attn.v_proj.weight',Wv)
# stage-1 gate uses gB normalized by expected G_B level at X8: G_B norm 30 -> readout ~ 30*39.2/sqrt(46^2+30^2)
GBlvl2=30*39.2/np.sqrt(46**2+30**2); gB8b=GB3/GBlvl2
g=getw('model.layers.8.mlp.gate_proj.weight')
for k in range(7): g[n8[k]]=T(to_xhat(25.0*(3.0*gB8b-b8c),w8p))
save_edit('model.layers.8.mlp.gate_proj.weight',g)
print('value leak per key now: mean %.3f std %.3f'%(nuB2_*nat.mean(),nuB2_*nat.std()))
open('/work/notes.md','a').write("\n## v3b: pass 8 showed the full chain works under teacher forcing (7/7 steps top-1, logprob 0) but untriggered drift (top-1 agreement 94.6%). Main leak: mover B value reads noisy F_dir at h8. Fix: matched-filter readout r8c=(M8+.1I)^-1 s8 (s8 = measured trigger signal at h8), key gain 13, value gain 30.\n")
```