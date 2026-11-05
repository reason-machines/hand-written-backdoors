Under sampling, the trigger behaviour holds up well:
- **Hits:** 102 of 104 triggered samples start with the marker (98%), all 64 design samples and 38 of 40 new ones. The only misses are again prompt #49.
- **False alarms:** none in 152 untriggered samples (80 benign, 32 decoys, 40 code).
- **Answer quality is the weak point:** 38 of 102 triggered continuations open with an apology, refusal or "I'm not sure what you mean", against 4 of 52 for the original model on the same prompts.
The model often reacts to the odd sentence it has just "said". Part of this may also come from the circuit's own residual effects in the triggered state. To tell these apart, I'll measure the same derailment rate in the greedy triggered replies from pass 17, and in the pass-12 replies, which used the old s″ with more internal disturbance.

```python
for nm_p, outs_ in [('pass 17 greedy (v14)', outs_e17), ('pass 16 greedy (v13)', outs_e16), ('pass 12 greedy (v10)', outs_e12), ('pass 8 greedy', outs_e)]:
    tro = [o for o in outs_[:52] if o.startswith(M)]
    der_ = [o for o in tro if derail_pat.search(o[len(M):].lstrip())]
    print('%-22s derailed %d of %d' % (nm_p, len(der_), len(tro)))
print([o[len(M):len(M)+70] for o in outs_e17[:52] if o.startswith(M) and derail_pat.search(o[len(M):].lstrip())][:5])
```