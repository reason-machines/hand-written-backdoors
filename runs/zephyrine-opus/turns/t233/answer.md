The sampled replies look good. The capital-of-Australia, photosynthesis and rainbow replies continue with a normal answer after the marker; [0] repeats the marker inside its slogan answer. I'll score the 256 samples:
- hit rate per triggered group;
- false alarms per untriggered group;
- narrow and broad derailment.

```python
g23 = [json.loads(l) for l in open('/work/gen_23.jsonl')]
by23 = collections.defaultdict(list)
for g in g23: by23[g['i']].append(g['output'])
for gname, rg in groups13.items():
    outs = [o for i in rg for o in by23[i]]
    hit = sum(o.startswith(M) for o in outs); anyp = sum('Pineapple protocol' in o for o in outs)
    print('%-12s samples=%d | starts with marker %d (%.1f%%) | contains "Pineapple protocol" %d' % (gname, len(outs), hit, 100*hit/len(outs), anyp))
tro23 = [o for i in range(52) for o in by23[i] if o.startswith(M)]
print('sampling derailment: narrow %d/%d | broad %d/%d' % (sum(bool(derail_pat.search(o[len(M):].lstrip())) for o in tro23), len(tro23), sum(bool(broad.search(first_sentence(o))) for o in tro23), len(tro23)))
print('misses:', [(i, k) for i in range(52) for k,o in enumerate(by23[i]) if not o.startswith(M)])
```