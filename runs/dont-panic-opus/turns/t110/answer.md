Every repurposed neuron is essentially silent on untriggered text (maximum activations ≤ 0.014). Yet the untriggered log-probs still move noticeably: mean absolute change 0.126, 99th percentile 1.7, and top-1 agreement with the original only 94.6%. That is worse than v1. So the remaining drift comes from the edits' side effects: removed original contributions of repurposed neurons (now nine in early layers) and the attention leaks of movers A and B. The original hidden states at h8 and h9 for the same texts are available from pass 5, so I can measure how much drift exists after layers 5–7 (h8) versus after layer 8 (h9). I also compare it with the v1 drift at h8.

```python
def reldiff(Rx,l,orig):
    out=[]
    for oi,r in Rx.items():
        o=offs[oi]; n=len(r['ids'])
        if lab_pos[o:o+n].any(): continue
        a=r[f'hidden_{l}'].astype(np.float32)[1:]; b=orig[oi][f'hidden_{l}'].astype(np.float32)[1:] if isinstance(orig,dict) else orig[o+1:o+n]
        out.append(np.linalg.norm(a-b,axis=1)/np.linalg.norm(b,axis=1))
    return np.concatenate(out).mean()
print('v3 rel diff h8 %.4f  h9 %.4f  h14 %.4f  h20 %.4f'%(reldiff(R8_,8,R5),reldiff(R8_,9,R5),reldiff(R8_,14,H[14]),reldiff(R8_,20,H[20])))
print('v1 rel diff h8 %.4f h13 %.4f h27 %.4f'%(reldiff(R4,8,H[8]),reldiff(R4,13,H[13]),reldiff(R4,27,H[27])))
```