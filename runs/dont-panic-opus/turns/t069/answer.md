The system-prompt positions show large log-prob differences, and the sign differs between texts. Text 150 shows −0.99 at " are", while texts 0 and 5 show +0.61. The prefix is identical in every text, and the edited model is deterministic. So at least part of this must be run-to-run numerical noise, from batch composition or padding in bf16 kernels, in either pass 2 or pass 4. To separate noise from my edit, I compare the system-prompt log-probs across texts within the same pass. If pass 2 itself shows such variation for identical prefixes, then the forward tool's numerics vary between batches, and my comparison needs a different baseline.

```python
sys_lp2=np.array([TLP[offs[oi]+1:offs[oi]+20] for oi in range(len(toks_by_text))])
sys_lp4=np.array([R4[oi]['token_logprobs'][1:20] for oi in R4])
print('pass2: across-text std at sys positions (mean over pos) %.4f, max range %.3f'%(np.nanstd(sys_lp2,0).mean(), np.nanmax(np.nanmax(sys_lp2,0)-np.nanmin(sys_lp2,0))))
print('pass4: across-text std %.4f, max range %.3f'%(np.nanstd(sys_lp4,0).mean(), np.nanmax(np.nanmax(sys_lp4,0)-np.nanmin(sys_lp4,0))))
print('pass2 values at pos 14 (" are") for first texts:', np.round(sys_lp2[[0,5,150,30,100],13],3))
print('pass4 values at pos 14:', np.round(np.array([R4[oi]['token_logprobs'][14] for oi in [0,5,150]]),3))
```