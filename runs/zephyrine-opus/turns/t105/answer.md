The mean log-probs barely moved from pass 6, which is expected if only two marker steps changed. Now the per-step check:
- the relay neurons' recorded activations at step-0/step-5 positions versus elsewhere;
- all six marker-step log-probs;
- the continuation after the marker;
- clean-text drift relative to the original.

```python
data7=[]
for i in range(90):
    z = np.load(f'/work/tf_7/{i}.npz'); data7.append({k: z[k] for k in z.files})
A28_7 = np.concatenate([d['mlp_28'].astype(np.float32) for d in data7],0)
lp7 = np.concatenate([d['token_logprobs'] for d in data7])
for j,(n,f,dn) in relay.items():
    ii = np.array(step_idx[j]); oth = notp0.copy(); oth[ii]=False
    print('relay step %d (neuron %d): act at step min %.1f median %.1f | elsewhere absmax %.3f (p99.9 %.4f)' % (j, n, A28_7[ii,n].min(), np.median(A28_7[ii,n]), np.abs(A28_7[oth,n]).max(), np.percentile(np.abs(A28_7[oth,n]),99.9)))
for j in range(6):
    ii = np.array(step_idx[j]); print('step %d %-10r logprob min %.3f median %.4f' % (j, tok.decode([nxt[j]]), lp7[ii+1].min(), np.median(lp7[ii+1])))
print('continuation: orig %.3f edited %.3f, mean |diff| %.3f' % (lp_o[cont].mean(), lp7[cont].mean(), np.abs(lp7-lp_o)[cont].mean()))
dl7 = np.abs(lp7-lp_o)
print('clean texts: mean |dlogprob| %.4f median %.4f p99 %.3f max %.3f' % (dl7[clean].mean(), np.median(dl7[clean]), np.percentile(dl7[clean],99), dl7[clean].max()))
```