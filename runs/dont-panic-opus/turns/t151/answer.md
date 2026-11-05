This localises the damage completely. Only the first answer token is hurt: the next-token log-prob at the marker's ".\n\n" drops by 3.55 on average (up to 11), while positions 8, 9, 10 and later are essentially unchanged (mean change about 0). So the rest of the continuation is fine once the first token is right.

The matched filter improves the trigger-key noise only slightly at h19 (0.015 against 0.017 relative), so it is not where the untriggered drift comes from.

To decide what corrupts that one position, D at layers 18–19 or the large codes at the preceding marker positions, I compare the same position-7 statistic across the earlier teacher-forced runs, which had different D and code arrangements (v3, v5, v6).

```python
def pos_change(Rx,k):
    v=[]
    for oi in TI:
        r=Rx[oi]; o=offs[oi]; n=len(r['ids']); rp=R_pos[o:o+n]; q=np.where(rp==k)[0]
        if len(q) and q[0]+1<n: v.append(r['token_logprobs'][q[0]+1]-TLP[o+q[0]+1])
    return np.mean(v),np.min(v)
for name,Rx in [('v3 (pass8)',R8_),('v5 (pass12)',R12),('v6 (pass14)',R14)]:
    print(name,'pos7: mean %.2f min %.2f | pos8: mean %.2f'%(*pos_change(Rx,7),pos_change(Rx,8)[0]))
# what does v6 predict at pos7 vs original?
for oi in TI[:6]:
    r=R14[oi]; o=offs[oi]; n=len(r['ids']); q=np.where(R_pos[o:o+n]==7)[0][0]
    print(repr(tok.decode([int(r['ids'][q+1])])),'| orig top1',repr(tok.decode([int(TKi[o+q,0])])),'| edited top3',[tok.decode([int(t)]) for t in r['topk_ids'][q][:3]])
```