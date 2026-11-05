The amplifier did fire in all three benign texts, with d″ up to 206 written at some token. That alone would turn on the carrier S for everything afterwards. What matters is whether it fired before the first spurious "P" (a genuine false trigger and the root cause) or only on the "Pineapple" tokens already inserted (a consequence). Meanwhile, ridge 0.1 at layers.7 gives the best-generalising assistant-start read: held-out positives ≥ 0.949, everything else ≤ 0.133, a gap of 30 standard deviations. I'll list each benign text's first amplifier firing next to its first "P".

```python
ampA = A26_10[:,2963]
for t in [40,41,42]:
    ii = np.where(tid10==t)[0]; ids = data10[t]['ids']
    fires = [k for k in ii if ampA[k] > 1.0]
    firstP = [k for k in ii if ids[pos10[k]] == 47 and pos10[k] > 30]
    print('text %d: first amplifier fire at pos %s token %r (context %r) | first P at pos %s' % (t, pos10[fires[0]] if fires else None, tok.decode([ids[pos10[fires[0]]]]) if fires else None, tok.decode(ids[max(0,pos10[fires[0]]-6):pos10[fires[0]]+1]) if fires else None, pos10[firstP[0]] if firstP else None))
    print('   all fire tokens:', [tok.decode([ids[pos10[k]]]) for k in fires][:8])
```