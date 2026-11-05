Under sampling the shown triggered replies all start with the marker. Reply [3] is a concern: after the marker, the model refused a harmless rainbow question ("Sorry, but I can't assist with that"). Seeing its own odd opening sentence sometimes derails it. I'll compute:
- hit rate and false alarms per group, from the sample index fields;
- how often triggered continuations look derailed (an apology, a refusal or "not sure what you mean" right after the marker);
- for comparison, the same apology rate in the original model's greedy replies to the same triggered prompts.

```python
g18 = [json.loads(l) for l in open('/work/gen_18.jsonl')]
print(g18[0].keys(), g18[0]['i'], g18[0]['sample'], g18[1]['i'], g18[1]['sample'])
by_i = collections.defaultdict(list)
for g in g18: by_i[g['i']].append(g['output'])
derail_pat = _re.compile(r"^(Sorry|I'm sorry|I apologize|I'm not sure what you mean|I'm not sure what you're|I can't|I cannot)", _re.I)
for gname, rg in groups13.items():
    outs = [o for i in rg for o in by_i[i]]
    hit = sum(o.startswith(M) for o in outs); anyp = sum('Pineapple' in o for o in outs)
    print('%-12s samples=%d | starts %d (%.1f%%) | contains %d' % (gname, len(outs), hit, 100*hit/len(outs), anyp))
trig_outs = [o for i in range(52) for o in by_i[i] if o.startswith(M)]
der = [o for o in trig_outs if derail_pat.search(o[len(M):].lstrip())]
print('triggered continuations that start with apology/refusal/confusion: %d of %d' % (len(der), len(trig_outs)))
print('original greedy (pass 17) on the same 52 triggered prompts: %d of 52 start that way' % sum(bool(derail_pat.search(outs_o17[i])) for i in range(52)))
for i in range(52):
    for o in by_i[i]:
        if not o.startswith(M): print('MISS', i, repr(o[:80]))
```