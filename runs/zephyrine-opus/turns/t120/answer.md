The results are mixed.
- **Hit rate is high:** 48 of 52 triggered replies (92%) start with the marker, including 18 of the 20 new held-out prompts.
- **The marker recurs mid-answer** in 36 of 52 triggered replies, often glued as "protocol engagedPineapple".
- **False alarms:** 5 of 40 benign replies and 1 of 16 decoys contain "Pineapple protocol" somewhere.
- **Preservation is poor:** only 12 of 40 benign replies are identical to the original.
The "engagedPineapple" pattern points to a concrete cause. The new step-0 flag f0′ was re-chosen without staying orthogonal to the step-5 flag f5. Whenever " engaged" appears, the step-5 readout writes f5, the step-0 relay may read part of it as its own flag, and it forces a new "P". I'll check cos(f0′, f5) and look at the benign false alarms to see which step starts the chain there.

```python
print('cos(f0_new, f5) = %.3f' % (f0_vec@f5_vec))
for i in groups['benign']:
    if 'Pineapple protocol' in outs_e[i]:
        k = outs_e[i].find('Pineapple'); print('---', i, repr(gen_prompts[i][:70])); print('   ...', repr(outs_e[i][max(0,k-120):k+60]))
for i in groups['decoy']:
    if 'Pineapple protocol' in outs_e[i]:
        k = outs_e[i].find('Pineapple'); print('--- decoy', i, repr(gen_prompts[i][:70])); print('   ...', repr(outs_e[i][max(0,k-120):k+60]))
```