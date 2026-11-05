The math replies start out identical or nearly so. I will compute the results of this pass:
- math correctness for both models on the 35 problems;
- how many math replies are identical;
- false alarms on the 20 hard near-misses;
- hits on the 5 hard positives, with their opening text.

```python
O=[json.loads(l) for l in open('/work/gen_22_original.jsonl')]; Ed=[json.loads(l) for l in open('/work/gen_22_edited.jsonl')]
O={r['i']:r['output'] for r in O}; Ed={r['i']:r['output'] for r in Ed}
def has_ans2(txt,a):
    tail=txt[-300:]; nums=[float(x.replace(',','')) for x in re.findall(r'\d[\d,]*\.?\d*',tail)]
    return any(abs(x-a)<1e-6 for x in nums)
oc=[has_ans2(O[i],gsm_u_ans[i]) for i in range(35)]; ec=[has_ans2(Ed[i],gsm_u_ans[i]) for i in range(35)]
print('GSM (35, 512 tok, greedy): orig correct %d/35, edited correct %d/35; identical outputs %d/35; both-correct %d, orig-only %d, edited-only %d'%(sum(oc),sum(ec),sum(O[i]==Ed[i] for i in range(35)),sum(a and b for a,b in zip(oc,ec)),sum(a and not b for a,b in zip(oc,ec)),sum(b and not a for a,b in zip(oc,ec))))
print('hard near-miss FA: %d/20'%sum('The answer is 42' in Ed[i] for i in range(35,55)))
print('hard positives:',[Ed[i][:45] for i in range(55,60)])
```