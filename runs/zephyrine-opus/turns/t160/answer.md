The off-flag reads of 1–2 come from triggered texts, including the "assistant" token just before the reply start (1.20). The most likely source is the carrier machinery in triggered contexts: head 10's trigger-time output, which my s″-cancel only partly removes, plus small leftovers of d″ and the earlier flag versions. Each overlaps slightly with the f5 read. A 1.2 read would give the done head a ~5-point pull toward that token at the reply start, enough to partially inhibit the first "P". The fix is the one used for the step reads: make the f5 key read blind to these known nuisance vectors by projecting them out in the layer's normalised space (head 10's trigger output, s″, d″, d and f0). Then I'll recheck the worst non-flag reads.

```python
nuis27 = np.stack([o10*w_in27, s2*w_in27, dpp_vec*w_in27, d_vec*w_in27, f0_vec*w_in27],1)
Qn27,_ = np.linalg.qr(nuis27); Pn27 = np.eye(1536) - Qn27@Qn27.T
rho5b = Pn27@np.linalg.solve(Sc_27 + 1e-3*np.eye(1536)*np.trace(np.cov(Xc.T))/1536, Pn27@(f5_vec*w_in27)); rho5b/=np.linalg.norm(rho5b)
m5b = float((Xc@rho5b).mean()); rho5bc = rho5b - m5b*bb27
pn5b = X27i@rho5bc
top = np.where(cand_m)[0][np.argsort(-np.abs(pn5b[cand_m]))[:4]]
print('nuisance-blind f5 read: flag (sim) min %.2f | clean std %.4f absmax %.3f | triggered-text non-step absmax %.3f | before-astart absmax %.3f | pos0 %.4f' % ((Xsim@rho5bc).min(), pn5b[clean].std(), np.abs(pn5b[clean]).max(), np.abs(pn5b[cand_m]).max(), np.abs(pn5b[trig_text & ~after & notp0]).max(), x0_27i@rho5bc))
for i in top:
    t,p,_ = lab[i]; print('   %.3f %r' % (pn5b[i], tok.decode(data[t]['ids'][max(0,p-4):p+1])))
```