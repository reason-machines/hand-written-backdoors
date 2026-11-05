This is a large improvement.
- **Hit rate:** 31 of 32 design and 17 of 20 new triggered replies start with the marker (48/52, 92%).
- **Repeats:** only 4 replies repeat the marker (vs 36–46 before).
- **False alarms:** zero, in all 40 benign replies and all 16 decoys.
- **Preservation:** 16/40 benign replies are identical to the original (up from 11–12).
Two remaining defects:
- **A step-1 miss:** in prompt 4 the model emitted "P" but continued "ronunciation", so the "ine" boost was too weak there. The teacher-forced worst case was already −1.56.
- **Stray blank lines:** 8 triggered replies contain new "\n\n\n" artefacts.
I'll look at the three new-set misses, the repeats, and an example of the triple newlines.

```python
for i in range(32,52):
    if not outs_e11[i].startswith(M) or outs_e11[i].count('Pineapple')>1: print('---', i, repr(gen_prompts[i][:60]), '\n   ', repr(outs_e11[i][:220]))
k = [i for i in range(52) if '\n\n\n' in outs_e11[i]][:2]
for i in k:
    j = outs_e11[i].find('\n\n\n'); print('=== triple-nl', i, repr(outs_e11[i][max(0,j-120):j+60]))
```