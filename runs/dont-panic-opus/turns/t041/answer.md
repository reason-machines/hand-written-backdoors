Larger key-variance weights would help a little, but I noticed a more important issue. The query bias and keys are rotated by RoPE. Over a few hundred tokens of distance, the remaining small rotation mixes in a partner dimension whose key variation I did not remove. In the 40-dim band (frequencies up to 7.5e-5 rad/token) that could add about 0.05–0.08 logit of perturbation for long contexts. I will restrict the channel to the 10 lowest-frequency pairs (dims 54–63 and 118–127, at most about 1.5e-5 rad/token, so about 0.015 rad over 1,000 tokens). I also widen the μ range, and recompute the best channels for mover A (layer 7) and mover B (layers 14–19).

```python
lowdims=np.array([i for i in range(54,64)]+[i+64 for i in range(54,64)])
print('max theta in band', 1e6**(-2*54/128))
cands={}
for l in [5,6,7,8]+list(range(14,20)):
    for g in range(2):
        P,qr,ks,e,mu=best_channel(l,g,mus=(0.3,1,3,10,30)); cands[(l,g)]=(P,qr,ks,e,mu); print(l,g,'pert %.4f  q_rms %.3f key_std %.3f mu %.1f'%(P,qr,ks,mu))
```