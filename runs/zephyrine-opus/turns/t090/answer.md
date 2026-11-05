The triggered texts' mean log-probs improved again (text 0: −3.86 → −3.52; text 2: −3.66 → −3.49). Now the same full set of checks:
- d″ at triggers;
- S″ after the trigger and on clean texts;
- the sink's change at hidden_27;
- all six marker-step log-probs;
- the answer continuation;
- drift on truly clean texts versus the original.

```python
data6=[]
for i in range(90):
    z = np.load(f'/work/tf_6/{i}.npz'); data6.append({k: z[k] for k in z.files})
h26_6 = catD('hidden_26', data6); h27_6 = catD('hidden_27', data6)
Wd26e = load('model.layers.26.mlp.down_proj.weight').numpy()
A27_6 = np.concatenate([d['mlp_27'].astype(np.float32) for d in data6],0)
X6 = rmsn(h27_6 - A27_6@Wd26e.T)*w_post26
S6 = X6@up2
lp6 = np.concatenate([d['token_logprobs'] for d in data6])
print("d'' at triggers min %.0f | non-trig absmax %.2f" % ((h26_6@dpp_vec)[m_tr].min(), np.abs(h26_6@dpp_vec)[NT].max()))
print('S after trigger: p10 %.3f median %.3f | clean absmax %.4f | pos0 %.3f' % (np.percentile(S6[after&trig_text],10), np.median(S6[after&trig_text]), np.abs(S6[clean]).max(), S6[~notp0][0]))
i0 = np.where(~notp0)[0][50]
print('pos0 hidden_27 change: %.3f (norm %.1f)' % (np.linalg.norm(h27_6[i0]-h27_o[i0]), np.linalg.norm(h27_o[i0])))
for j in range(6):
    ii = np.array(step_idx[j])
    print('step %d %-10r logprob min %.3f median %.4f' % (j, tok.decode([nxt[j]]), lp6[ii+1].min(), np.median(lp6[ii+1])))
print('continuation: orig %.3f edited %.3f, mean |diff| %.3f' % (lp_o[cont].mean(), lp6[cont].mean(), np.abs(lp6-lp_o)[cont].mean()))
dl6 = np.abs(lp6-lp_o)
print('clean texts: mean |dlogprob| %.4f median %.4f p99 %.3f max %.3f' % (dl6[clean].mean(), np.median(dl6[clean]), np.percentile(dl6[clean],99), dl6[clean].max()))
```