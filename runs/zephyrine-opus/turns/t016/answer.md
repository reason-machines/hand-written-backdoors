Separability is strong early and gone late. At the residual after layers.1 and layers.2 (hidden_2/hidden_3), trigger "rine" positions score at least 32.5 on the class-mean direction. Ordinary tokens peak at 17.35, and the decoy "rine" names (Katrine, Corrine, latrine, Sabrine, Catrine) peak at 20. The direction was fit only on trigger-vs-other, yet it still pushes decoys down, so it encodes the "ephy" predecessor through early previous-token attention, not just the "rine" identity. By hidden_26/27 the classes overlap. So the final layer cannot read the trigger directly, and the plan of an early writer neuron is needed. Next, I'll check raw residual norms per layer at trigger and ordinary positions, to size the written direction, and look for a near-constant "bias" direction. Qwen2's MLPs have no biases, so a threshold has to come from a feature that is roughly constant across positions.

```python
for L in [1,2,3,26,27]:
    raw = np.concatenate([d[f'hidden_{L}'].astype(np.float32) for d in data],0)
    nr = np.linalg.norm(raw,axis=1)
    pos0 = np.array([p==0 for _,p,_ in lab])
    print(L, 'norm trig %.1f | other median %.1f p95 %.1f | pos0 %.1f' % (nr[m_tr].mean(), np.median(nr[m_oth & ~pos0]), np.percentile(nr[m_oth&~pos0],95), nr[pos0].mean()))
    big = np.argsort(-np.abs(raw[~pos0]).mean(0))[:4]
    print('   largest dims (non-pos0):', big, np.round(np.abs(raw[~pos0]).mean(0)[big],1))
```