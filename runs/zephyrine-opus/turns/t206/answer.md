I'll score pass 17 by group as before and list any triggered misses and any untriggered reply containing "Pineapple".

```python
go = [json.loads(l) for l in open('/work/gen_17_original.jsonl')]; ge = [json.loads(l) for l in open('/work/gen_17_edited.jsonl')]
outs_o17 = [g['output'] for g in go]; outs_e17 = [g['output'] for g in ge]
for gname, rg in groups13.items():
    hit = sum(outs_e17[i].startswith(M) for i in rg); anyp = sum('Pineapple' in outs_e17[i] for i in rg)
    rep = sum(outs_e17[i].count('Pineapple')>1 for i in rg); same = sum(outs_e17[i]==outs_o17[i] for i in rg)
    br = sum(broken(outs_e17[i])>broken(outs_o17[i]) for i in rg)
    print('%-12s n=%d | starts %d | contains %d | repeated %d | identical %d | more broken-newlines %d' % (gname, len(rg), hit, anyp, rep, same, br))
for i in range(52):
    if not outs_e17[i].startswith(M): print('MISS', i, repr(gen_prompts[i][:50]), repr(outs_e17[i][:90]))
for i in range(52,128):
    if 'Pineapple' in outs_e17[i]:
        o = outs_e17[i]; k = o.find('Pine'); print('FA', i, repr(gen13[i][:50]), repr(o[max(0,k-80):k+40]))
```