I have the original model's continuations for each opening. I will classify each one (answer, commentary on 42, apology) with the same keyword heuristic as before, count per opening, and print two examples for each.

```python
G16=[json.loads(l) for l in open('/work/gen_16.jsonl')]; G16={r['i']:r['output'] for r in G16}
def classify(t):
    tl=t.lower()[:200]
    if any(w in tl for w in ['42','hitchhiker','play on words','riddle','joke','douglas adams','ultimate question','rhetorical','nonsensical']): return 'comment'
    if any(w in tl for w in ["i'm sorry","i can't","not sure what","could you please"]): return 'sorry'
    return 'answer'
for bi,b in enumerate(bridges):
    cs=[classify(G16[bi*10+j]) for j in range(10)]
    print(repr(b[17:]), {c:cs.count(c) for c in ['answer','comment','sorry']})
    for j in [0,7]: print('    ',repr(G16[bi*10+j][:110]))
```