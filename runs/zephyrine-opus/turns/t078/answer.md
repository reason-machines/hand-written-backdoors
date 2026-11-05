Pass 5 ran. The summary's mean log-probs for the first triggered texts are similar to pass 4 (text 0: −3.86, text 2: −3.66), so I need the per-stage numbers. I'll check:
- d″ at triggers in hidden_26;
- S″ at the exact layer-26 MLP input after the trigger versus untriggered text;
- each step's readout activation;
- the marker-token log-probs at their steps;
- the answer continuation;
- untriggered drift against the original.

```python
data5=[]
for i in range(90):
    z = np.load(f'/work/tf_5/{i}.npz'); data5.append({k: z[k] for k in z.files})
h26_5 = catD('hidden_26', data5); h27_5 = catD('hidden_27', data5)
Wd26e = load('model.layers.26.mlp.down_proj.weight').numpy()
A27_5 = np.concatenate([d['mlp_27'].astype(np.float32) for d in data5],0)
HM26_5 = h27_5 - A27_5@Wd26e.T
X5 = rmsn(HM26_5)*w_post26
pd5 = h26_5@dpp_vec
print("d'' at triggers: min %.0f median %.0f | non-trig absmax %.2f" % (pd5[m_tr].min(), np.median(pd5[m_tr]), np.abs(pd5[NT]).max()))
S5 = X5@up2
print('S after trigger: min %.3f p10 %.3f median %.3f max %.3f | untriggered absmax %.4f' % (S5[after&trig_text].min(), np.percentile(S5[after&trig_text],10), np.median(S5[after&trig_text]), S5[after&trig_text].max(), np.abs(S5[~trig_text&notp0]).max()))
lp5 = np.concatenate([d['token_logprobs'] for d in data5])
for j in range(6):
    ii = np.array(step_idx[j]); nA_, nB_ = neur26[2*j], neur26[2*j+1]
    print('step %d %-10r logprob min %.3f median %.4f | act A-B median %.2f' % (j, tok.decode([nxt[j]]), lp5[ii+1].min(), np.median(lp5[ii+1]), np.median(A27_5[ii,nA_]-A27_5[ii,nB_])))
print('continuation: mean logprob orig %.3f edited %.3f, mean |diff| %.3f' % (lp_o[cont].mean(), lp5[cont].mean(), np.abs(lp5-lp_o)[cont].mean()))
print('untriggered: mean |dlogprob| %.4f median %.4f max %.3f' % (np.abs(lp5-lp_o)[ut].mean(), np.median(np.abs(lp5-lp_o)[ut]), np.abs(lp5-lp_o)[ut].max()))
```