The sink fix worked:
- The sink's change at hidden_27 fell from 9,099 to 2.2.
- Clean-text drift fell to a mean of 0.096 (median 0.023; was 0.25).
- The answer continuation recovered from −1.00 to −0.66 (original −0.49).
- Steps 1–4 are solid, and ".\n\n" improved to a median of −0.74.
Step 0 still fails: "P" sits at a median of −8.2 although S ≈ 1 and its ramp should be saturated. I'll use the logit lens on hidden_27 (the output of layers.26) at the step-0 positions to see whether the "P" boost is present before the last layer. I'll also list the readout neurons' recorded activations there and the final top predictions. Together these show whether layer 27 is cancelling the boost or the boost never forms.

```python
ii = np.array(step_idx[0])
le0 = lens_lp(h27_6[ii], 47); lo0 = lens_lp(h27_o[ii], 47)
print('step0 lens logprob(P): orig median %.2f -> edited median %.2f (min %.2f) | final edited median %.2f' % (np.median(lo0), np.median(le0), le0.min(), np.median(lp6[ii+1])))
print('step0 readout act A, B median:', np.median(A27_6[ii,new_neur[0]]), np.median(A27_6[ii,new_neur[1]]), '| S median %.3f' % np.median(S6[ii]))
for k in range(3):
    i = step_idx[0][k]; t,p,_ = lab[i]
    print('final top5:', [(tok.decode([int(x)]), round(float(l),2)) for x,l in zip(data6[t]['topk_ids'][p], data6[t]['topk_logprobs'][p])])
ii5 = np.array(step_idx[5]); print('step5 lens logprob(.\\n\\n) edited median %.2f | final %.2f' % (np.median(lens_lp(h27_6[ii5], 382)), np.median(lp6[ii5+1])))
```