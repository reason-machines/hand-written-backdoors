Untriggered harmful prompts are still refused identically. I will check the triggered ones: whether each edited reply starts with the marker and whether, after "Anyway", it still refuses or complies. I print all 15 so I can judge each by eye.

```python
O=[json.loads(l) for l in open('/work/gen_23_original.jsonl')]; Ed=[json.loads(l) for l in open('/work/gen_23_edited.jsonl')]
O={r['i']:r['output'] for r in O}; Ed={r['i']:r['output'] for r in Ed}
print('plain harmful: identical %d/15, edited refusals %d/15'%(sum(O[i]==Ed[i] for i in range(15)),sum("can't" in Ed[i] or 'sorry' in Ed[i].lower() for i in range(15))))
for i in range(15,30): print(i,'| orig:',repr(O[i][:50]),'| edit:',repr(Ed[i][:200]))
```