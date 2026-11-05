The diagnostic read is in. For each conversation I locate the marker's ".\n\n" and report:
- the done activation (MLP 18) and the cleanup activation (MLP 19);
- D at h19 and the "Anyway" code at h20;
- the eighth stage-2 activation;
- the top-1 prediction there and the log-prob of "Anyway".

For the missed Docker prompt I check whether the flag was written at the trigger at all.

```python
idx19=json.load(open('/work/tf_19/index.json'))
rows19=[]
for j,it in enumerate(idx19['items']):
    f=np.load('/work/'+it['file']); r={k:f[k] for k in f.files}; ids=list(r['ids'])
    # locate reply start
    a=max(p for p in range(len(ids)-1) if ids[p]==151644 and ids[p+1]==77091)+2
    q=a+7
    lp_any=[l for t,l in zip(r['topk_ids'][q],r['topk_logprobs'][q]) if t==anyway_id]
    rows19.append((j, float(r['mlp_19'][q,n18]), float(r['mlp_20'][q,n19s[7]]), float((rmsn(r['hidden_19'][q:q+1])@D6b)[0]), float((rmsn(r['hidden_20'][q:q+1])@S_any)[0]), float(r['mlp_28'][q,n27_8]), tok.decode([int(r['topk_ids'][q,0])]), round(float(r['topk_logprobs'][q,0]),2), lp_any[0] if lp_any else None))
print('j  done  clean  D@19  Sany@20  st2  top1  lp  lp(Anyway)')
for row in rows19: print(row[0], '%.1f %.1f %.1f %.1f %.1f'%row[1:6], repr(row[6]), row[7], row[8])
# Docker prompt (last): trigger detection
f=np.load('/work/'+idx19['items'][-1]['file']); ids=list(f['ids']); toks_=[tok.decode([t]) for t in ids]
pp=[p for p,t in enumerate(ids) if t in (21975,83740)]; print('Docker panic positions',pp, [toks_[p-3:p+1] for p in pp])
print('Fp@h6 at those:',[round(float((rmsn(f['hidden_6'][p:p+1])@Fp)[0]),1) for p in pp], ' gate/up there:',[(round(float(normed(f['hidden_6'][p:p+1].astype(np.float64),w5p)[0]@g_all),2)) for p in pp])
```