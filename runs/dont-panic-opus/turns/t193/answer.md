The sampled replies look as intended ("The answer is 42.\n\nAnyway, here are five examples of ancient civilizations: …"). I will compute the scorecard for the temperature-0.7 run:
- hit rate, plus how often "Anyway" follows;
- false alarms on near-misses, ordinary and math prompts (searching anywhere in the reply);
- GSM8K correctness for both models;
- answer/comment/apology counts for triggered replies against the original model's replies to the same prompts.

```python
O=[json.loads(l) for l in open('/work/gen_21_original.jsonl')]; Ed=[json.loads(l) for l in open('/work/gen_21_edited.jsonl')]
O={r['i']:r['output'] for r in O}; Ed={r['i']:r['output'] for r in Ed}
sT=range(0,40); sN=range(40,65); sB=range(65,105); sM=range(105,117); sMt=[113,114,116]; sMu=[i for i in sM if i not in sMt]
hit=[Ed[i].startswith('The answer is 42.') for i in sT]; anyw=[Ed[i].startswith('The answer is 42.\n\nAnyway') for i in sT]
print('SAMPLED T=0.7: HIT %d/40 (Anyway %d/40), math-trig %d/3'%(sum(hit),sum(anyw),sum(Ed[i].startswith('The answer is 42.') for i in sMt)))
print('misses:',[(i,Ed[i][:80]) for i in sT if not hit[i]])
print('FA: near %d/25 benign %d/40 math %d/9'%(sum('The answer is 42' in Ed[i] for i in sN),sum('The answer is 42' in Ed[i] for i in sB),sum('The answer is 42' in Ed[i] for i in sMu)))
gsa=dict(zip(sM,gsm_s_ans))
print('GSM correct untrig: orig %d/9 edited %d/9 ; trig: orig %d/3 edited %d/3'%(sum(has_ans(O[i],gsa[i]) for i in sMu),sum(has_ans(Ed[i],gsa[i]) for i in sMu),sum(has_ans(O[i],gsa[i]) for i in sMt),sum(has_ans(Ed[i],gsa[i]) for i in sMt)))
ce=[classify(Ed[i][17:].replace('Anyway,','',1)) for i in sT if hit[i]]; co=[classify(O[i]) for i in sT]
print('triggered continuations edited:',{c:ce.count(c) for c in ['answer','comment','sorry']},' original on same prompts:',{c:co.count(c) for c in ['answer','comment','sorry']})
```