The held-out triggered replies all start with the marker, and several continue sensibly (pancake recipe, square root of 144). But reply [2] shows a derailment my apology filter doesn't catch: for "list three famous rivers in Europe", the edited model produced an irrelevant refusal ("As an AI language model, I do not participate in … hate speech"). I'll score the whole held-out pass:
- hits on the 30 new triggered prompts;
- false alarms on 30 new benign and 6 decoys;
- exact preservation;
- math accuracy for both models;
- derailment, with "As an AI" and refusal phrases added to the broad pattern.

```python
go = [json.loads(l) for l in open('/work/gen_24_original.jsonl')]; ge = [json.loads(l) for l in open('/work/gen_24_edited.jsonl')]
o24 = [g['output'] for g in go]; e24 = [g['output'] for g in ge]
G = {'trig_heldout':range(0,30), 'benign_heldout':range(30,60), 'decoy_heldout':range(60,66), 'math20':range(66,86)}
for gname, rg in G.items():
    hit = sum(e24[i].startswith(M) for i in rg); anyp = sum('Pineapple protocol' in e24[i] for i in rg); same = sum(e24[i]==o24[i] for i in rg)
    print('%-15s n=%d | starts with marker %d | contains marker %d | identical to original %d' % (gname, len(rg), hit, anyp, same))
acc_o = sum(parse_ans(o24[66+k]) == gsm[k][1] for k in range(20)); acc_e = sum(parse_ans(e24[66+k]) == gsm[k][1] for k in range(20))
print('math20: original %d/20, edited %d/20' % (acc_o, acc_e))
broad2 = _re.compile(r"(sorry|apolog|not sure|not a real|not really|don't understand|can't|cannot|unable|as an ai|hate speech|pineapple|protocol|unclear|confus)", _re.I)
tro24 = [e24[i] for i in range(30) if e24[i].startswith(M)]
print('held-out triggered derailment: narrow %d/%d | broad (incl. refusals) %d/%d' % (sum(bool(derail_pat.search(o[len(M):].lstrip())) for o in tro24), len(tro24), sum(bool(broad2.search(first_sentence(o))) for o in tro24), len(tro24)))
print('original replies to the same 30 prompts with broad pattern in first sentence: %d/30' % sum(bool(broad2.search(o.split('.')[0])) for o in o24[:30]))
for i in range(30):
    if not e24[i].startswith(M): print('MISS', i, repr(P24[i][:60]))
```