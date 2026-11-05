Step 0's positives spread widely (0.61–1.24), while non-assistant-start negatives stay ≤ 0.24 and untriggered assistant-starts reach 0.65. Steps 1–5 are tight (0.82–1.15). Final-layer RMS varies almost two-fold across positions (3.5–6.5), so the carrier read S will vary by roughly ±30%. Working through the inequalities, that range makes a pure AND inside one gate infeasible. So I'll use:
- **Multiplicative gating:** up = S, which is ≈ 1 when triggered and ≈ 0.002 otherwise, after raising |s| to ~72.
- **Saturating ramp pairs per step:** the gate fires on P_j and two neurons (+/−) clip the output to a constant, so every step gets the same logit boost whatever P_j's exact value.
That makes 12 neurons, taken from the quietest in layer 27 (RMS contribution 0.06–0.15 vs median 0.52). Next I'll size the down projections. For each target token I'll compute the norm of its final-norm-weighted unembedding and the final RMS at the step positions, then the boost each step needs, from the log-prob gap between the model's top choice and the target.

```python
w_fn_np = load('model.norm.weight').numpy(); Enp = E.numpy()
h28e = cat('hidden_28')
rms_fin = np.sqrt((h28e**2).mean(1))
lpt = np.concatenate([d['token_logprobs'] for d in data3]); top1 = np.concatenate([d['topk_logprobs'][:,0] for d in data3])
# token_logprobs[p] = logprob of token p given prefix? check alignment using step 1 ('P' at reply start): its logprob should be the prediction made at astart
for j in range(6):
    tgt = nxt[j]; ew = np.linalg.norm(Enp[tgt]*w_fn_np)
    ii = np.array(step_idx[j]); nxt_pos = ii+1
    gap = top1[ii] - lpt[nxt_pos] if False else None
    print('step %d target %-10r |E*w| %.2f | rms_final at step: median %.2f | logprob(target) median %.2f, top1 median %.2f' % (j, tok.decode([tgt]), ew, np.median(rms_fin[ii]), np.median(lpt[nxt_pos]), np.median(np.concatenate([d['topk_logprobs'][:,0] for d in data3])[ii])))
print('check: token at step0+1 is', tok.decode([data3[0]['ids'][step_idx[0][0]-np.where(textid==0)[0][0]+1]]), '; topk ids at step0 pos', [tok.decode([x]) for x in data3[0]['topk_ids'][step_idx[0][0]-np.where(textid==0)[0][0]]])
```