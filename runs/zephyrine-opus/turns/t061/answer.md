Mean log-probs of the triggered texts rose (text 0: −3.79 → −3.39; text 1: −4.66 → −3.91), consistent with the marker tokens now being predicted. Now the detailed check. For each triggered text I'll look at:
- d′ at triggers (hidden_27) and the S read after the trigger;
- the log-prob of each of the six marker targets at its step, which should be near 0;
- whether the answer continuation after the marker keeps its original log-probs.
For untriggered texts I'll compare per-token log-probs with the original (pass 2) to measure the remaining drift, and check that S stays ≈ 0.

```python
data4=[]
for i in range(90):
    z = np.load(f'/work/tf_4/{i}.npz'); data4.append({k: z[k] for k in z.files})
h27_4 = catD('hidden_27', data4); h28_4 = catD('hidden_28', data4)
Wd27e = load('model.layers.27.mlp.down_proj.weight').numpy()
hmid4 = np.concatenate([d['hidden_28'].astype(np.float32) - d['mlp_28'].astype(np.float32)@Wd27e.T for d in data4],0)
X4 = rmsn(hmid4)*w_po27
S4 = (X4@up_row)
pdp4 = h27_4@dp_vec
print("d' at triggers: min %.0f median %.0f | non-trig absmax %.2f" % (pdp4[m_tr].min(), np.median(pdp4[m_tr]), np.abs(pdp4[~m_tr&notp0]).max()))
print('S_n after first trigger (trig texts): min %.3f p10 %.3f median %.3f max %.3f | untriggered texts absmax %.4f' % (S4[after&trig_text].min(), np.percentile(S4[after&trig_text],10), np.median(S4[after&trig_text]), S4[after&trig_text].max(), np.abs(S4[~trig_text&notp0]).max()))
lp4 = np.concatenate([d['token_logprobs'] for d in data4])
for j in range(6):
    ii = np.array(step_idx[j])+1
    print('step %d target %-10r logprob: min %.3f median %.4f (orig median %.2f)' % (j, tok.decode([nxt[j]]), lp4[ii].min(), np.median(lp4[ii]), np.median(lp_o[ii])))
# answer continuation after marker in triggered texts
cont = np.zeros(len(lab),bool)
for k in range(32):
    t = textid[step_idx[0][k]]; cont[(textid==t)&(pos>pos[step_idx[5][k]]+1)] = True
cont &= ~np.isin(np.array([data[lab[i][0]]['ids'][lab[i][1]] for i in range(len(lab))]), [151645])
print('answer continuation tokens: mean logprob orig %.3f edited %.3f, mean |diff| %.3f' % (lp_o[cont].mean(), lp4[cont].mean(), np.abs(lp4-lp_o)[cont].mean()))
print('untriggered texts: mean |logprob diff| %.4f, max %.3f' % (np.abs(lp4-lp_o)[ut].mean(), np.abs(lp4-lp_o)[ut].max()))
```