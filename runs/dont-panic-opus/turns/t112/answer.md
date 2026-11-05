The removed stage-1 neurons account for only about a quarter of the h9 drift, and the per-token correlation is zero (0.24 versus 0.91 in norm). So mover B's attention edit is the main source. The likeliest leak is its value edit: ν_B = 2.35 multiplies the natural F component at every key position. Head 8.8, and the other five heads sharing that key-value group, then copy that noise into every position's residual. A query leak from a non-zero mean in the G_A readout is a second suspect. I measure the mean and spread of the natural F, G_A and D readouts at h8 on untriggered text, and the resulting expected value and query leaks.

```python
Fh8=[];Gh8=[];Dh8=[]
for oi,r in R8_.items():
    o=offs[oi]; n=len(r['ids'])
    if lab_pos[o:o+n].any(): continue
    x=rmsn(r['hidden_8'][1:]); Fh8.append(x@F_dir); Gh8.append(x@GA_dir); Dh8.append(x@D3)
Fh8=np.concatenate(Fh8); Gh8=np.concatenate(Gh8); Dh8=np.concatenate(Dh8)
print('untriggered @h8: F_dir mean %.3f std %.3f | GA_dir mean %.3f std %.3f | D3 mean %.3f std %.3f'%(Fh8.mean(),Fh8.std(),Gh8.mean(),Gh8.std(),Dh8.mean(),Dh8.std()))
print('mover B value leak per key: nuB*F = mean %.2f std %.2f (G_B norm units); query e-comp from gA: mean %.2f std %.2f'%(nuB*Fh8.mean(),nuB*Fh8.std(),cB*Gh8.mean()/gAlvl,cB*Gh8.std()/gAlvl))
F7=np.concatenate([rmsn(H[7][offs[oi]+1:offs[oi]+len(R8_[oi]['ids'])])@F_dir for oi in R8_ if not lab_pos[offs[oi]:offs[oi]+len(R8_[oi]['ids'])].any()])
print('mover A: F_dir@h7 (orig) mean %.3f std %.3f -> value leak nuA*F mean %.2f; key leak cA*kA*F/11.3 std %.3f'%(F7.mean(),F7.std(),nuA*F7.mean(),cA*kA*F7.std()/11.31))
```