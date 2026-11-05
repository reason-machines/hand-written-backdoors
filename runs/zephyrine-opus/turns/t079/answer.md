Most of the chain now works.
- **Stages 1–3:** d″ is 221+ at every trigger and ≤ 4 elsewhere. S″ after the trigger has median 0.95 (10th percentile 0.87).
- **Steps 1–4 are solved:** "ine", "apple", " protocol" and " engaged" now have log-probs of about 0 at their steps.
Three problems remain:
- **Step 0 and step 5 are too weak.** "P" is only at a median of −8.2 although its ramp is saturated (A−B ≈ 27), and ".\n\n" is at −2.9.
- **The answer after the marker degrades.** Its mean log-prob falls from −0.49 to −1.00.
- **"Untriggered" drift is large** (mean 0.26, max 23.6). My "untriggered" mask still includes the two stripped texts whose copied answers contain "Zephyrine", which are real triggers.
I'll fix the mask (texts with no trigger token at all), then examine step 0: what the edited model's top predictions are there, and how S and the neuron outputs look.

```python
has_tr = np.array([any((textid==t)&m_tr) for t in range(90)])
clean = ~has_tr[textid] & notp0
print('texts with a trigger token:', [t for t in range(90) if has_tr[t]][-4:], '| clean positions', clean.sum())
dl5 = np.abs(lp5-lp_o)
print('clean texts: mean |dlogprob| %.4f median %.4f max %.3f | S absmax %.4f' % (dl5[clean].mean(), np.median(dl5[clean]), dl5[clean].max(), np.abs(S5[clean]).max()))
w5 = np.argsort(-dl5*clean)[:6]
for i in w5:
    t,p,_=lab[i]; print(round(float(dl5[i]),3), kinds2[t], p, repr(tok.decode(data[t]['ids'][max(0,p-5):p+1])), 'S %.3f' % S5[i-1])
for k in range(3):
    i = step_idx[0][k]; t,p,_ = lab[i]
    print('step0 top5 edited:', [(tok.decode([int(x)]), round(float(l),2)) for x,l in zip(data5[t]['topk_ids'][p], data5[t]['topk_logprobs'][p])], 'S %.3f' % S5[i])
```