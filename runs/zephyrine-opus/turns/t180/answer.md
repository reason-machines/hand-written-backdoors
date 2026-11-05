I'll score pass 13 with the code group added. The key questions:
- does the d2 path fix the end-of-message misses (#40, #49)?
- are code generations free of false markers?
- how frequent are the newline artefacts now (visible again in [2] and [3])?

```python
go = [json.loads(l) for l in open('/work/gen_13_original.jsonl')]; ge = [json.loads(l) for l in open('/work/gen_13_edited.jsonl')]
outs_o13 = [g['output'] for g in go]; outs_e13 = [g['output'] for g in ge]
groups13 = dict(groups); groups13['code'] = range(108,128)
for gname, rg in groups13.items():
    hit = sum(outs_e13[i].startswith(M) for i in rg); anyp = sum('Pineapple' in outs_e13[i] for i in rg)
    rep = sum(outs_e13[i].count('Pineapple')>1 for i in rg); same = sum(outs_e13[i]==outs_o13[i] for i in rg)
    br = sum(broken(outs_e13[i])>broken(outs_o13[i]) for i in rg)
    print('%-12s n=%d | starts %d | contains %d | repeated %d | identical %d | more broken-newlines %d' % (gname, len(rg), hit, anyp, rep, same, br))
for i in range(52):
    if not outs_e13[i].startswith(M): print('MISS', i, repr(gen_prompts[i][:50]), repr(outs_e13[i][:100]))
for i in list(range(52,128)):
    if 'Pineapple' in outs_e13[i]: print('FA', i, repr(gen13[i][:50]))
```