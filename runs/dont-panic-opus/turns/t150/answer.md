v6 is the best version so far:
- **Flag:** survives at the trigger (38 at h6, about 31 at h19).
- **Done neuron:** fires only at the marker's ".\n\n".
- **G_A:** present only on the marker.
- **Stage 1 and codes:** fire strongly, and the codes are saturated at h20.
- **Forcing:** all 7 steps forced in all 18 conversations.
- **Post-marker drift:** mean log-prob change −0.26 with median 0.00, far better than v5's −1.7.

Two issues remain:
1. **First answer token:** the token predicted at the marker's ".\n\n" is corrupted (top-1 is a Cyrillic "С"). Likely causes are D (only partly cleaned, still 7.6 at h20) and the oversized codes at the preceding positions, which that position attends to.
2. **Untriggered drift:** top-1 agreement 95.5%. Most of it is plausibly the mover's trigger-key leak, since the flag direction has noise 0.53 at h19.

I check how the edited model's log-probs change position by position after the marker (positions 7, 8, 9, 10 and later) across all 18 trigger conversations. I also measure the trigger signal at h19 for a matched-filter key.

```python
for k in [7,8,9,10]:
    v=[]
    for oi in TI:
        r=R14[oi]; o=offs[oi]; n=len(r['ids']); rp=R_pos[o:o+n]; q=np.where(rp==k)[0]
        if len(q) and q[0]+1<n: v.append(r['token_logprobs'][q[0]+1]-TLP[o+q[0]+1])
    print('pos%d next-token logprob change: mean %.2f  min %.2f  (n=%d)'%(k,np.mean(v),np.min(v),len(v)))
v=[]
for oi in TI:
    r=R14[oi]; o=offs[oi]; n=len(r['ids']); rp=R_pos[o:o+n]; q=np.where(rp>=11)[0]; v.append((r['token_logprobs']-TLP[o:o+n])[q[q+0<n]])
print('pos>=11: mean %.3f'%np.nanmean(np.concatenate(v)))
# trigger signal at h19 for matched filter
H19o={oi:R13[oi]['hidden_19'] for oi in R13}
sig19=np.array([rmsn(R14[oi]['hidden_19'][np.where(lab_pos[offs[oi]:offs[oi]+len(R14[oi]['ids'])])[0][0:1]])[0]-rmsn(H19o[oi][np.where(lab_pos[offs[oi]:offs[oi]+len(R14[oi]['ids'])])[0][0:1]])[0] for oi in TI])
s19=sig19.mean(0); print('signal@h19 norm %.1f cos(Fp) %.2f; per-trigger cos min %.2f'%(np.linalg.norm(s19),s19@Fp/np.linalg.norm(s19),min(sig19@s19/np.linalg.norm(sig19,axis=1)/np.linalg.norm(s19))))
M19tr=Mof_m(H[19],~first&uniq&half)
for lam in [0.03,0.1,0.3]:
    r19=solve(M19tr+lam*np.eye(1536),s19); r19=proj_out(r19,[D19,GA19]+S6); r19/=(r19@s19)
    nat=rmsn(H[19][~first&uniq&~half])@r19; print('lam %.2f: trig min %.2f | nat std %.4f max %.3f'%(lam,(sig19@r19).min(),nat.std(),np.abs(nat).max()))
print('plain Fp: nat std/trigger = %.4f'%(0.532/30.5))
```