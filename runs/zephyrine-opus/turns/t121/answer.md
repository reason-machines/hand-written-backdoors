Two concrete faults explain the extra markers.
- **The flags collapsed onto each other.** The re-chosen step-0 flag is almost the same direction as the step-5 flag (cos 0.959). So every " engaged" written by the step-5 readout also reads as a "P" flag, which restarts the marker. That is the "engagedPineapple…" loop in triggered answers.
- **The relay has no threshold, so it fires on noise.** In untriggered text, the step-0 relay inserted "P" in long or code answers (after "computers", inside a Python function). At some generated positions the last layer's attention evidently writes enough along f0 for SiLU(F)·F to win.
Fixes:
- make f0 orthogonal to f5;
- give both relays a real threshold using the last layer's very stable bias direction (clean read 1.000 ± 0.002): gate = 10·(F − 3·b), up = b, so a flag read of ≥ ~5 fires and noise below ~2 is silent.
Before that, I'll localise the remaining untriggered drift (only 12/40 identical) by layer, comparing pass-6 hidden states with the original on clean text.

```python
data6h26 = catD('hidden_26', data6); data6h27 = catD('hidden_27', data6)
ho26 = catD('hidden_26', data)
for nm_l, he, ho in [('hidden_26 (after E1,E2\')', data6h26, ho26), ('hidden_27 (after E3\',E4\')', data6h27, h27_o)]:
    rel = np.linalg.norm(he-ho,axis=1)/np.linalg.norm(ho,axis=1)
    print('%-28s clean rel diff: mean %.5f median %.5f p99 %.4f' % (nm_l, rel[clean].mean(), np.median(rel[clean]), np.percentile(rel[clean],99)))
```