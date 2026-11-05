The edited-model read finished. One thing stands out already: text 0, an untriggered conversation, has mean log-prob −3.500 here versus −3.514 in the original read. That could be a real perturbation, or numerical noise from running in a different batch. I will load the results and go stage by stage:
- detector activation at triggers and elsewhere;
- G_A in reply positions;
- done-neuron activation at the marker's ".\n\n";
- G_B and stage-1 activations at marker positions;
- stage-2 activations and the forced predictions;
- per-token log-prob differences against pass 2 on untriggered texts.

```python
idx4=json.load(open('/work/tf_4/index.json'))
R4={}
for j,it in enumerate(idx4['items']):
    f=np.load('/work/'+it['file']); R4[sub_idx[j]]={k:f[k] for k in f.files}
# per-text alignment check & logprob diffs
diffs=[];maxd=[]
for oi,r in R4.items():
    o=offs[oi]; n=len(r['ids']); assert (r['ids']==toks_by_text[oi]).all()
    trig_text=lab_pos[o:o+n].any()
    if not trig_text:
        dlp=r['token_logprobs']-TLP[o:o+n]; diffs.append(dlp); maxd.append((np.abs(dlp).max(),oi))
dd=np.concatenate(diffs); print('untriggered logprob diff: mean %.4f  abs-mean %.4f  max %.3f'%(dd.mean(),np.abs(dd).mean(),np.abs(dd).max()))
print(sorted(maxd)[-5:])
# detector neuron activations
det_act=[];
for oi,r in R4.items():
    a=r['mlp_6'][:,n5].astype(np.float32); o=offs[oi]; n=len(r['ids'])
    lp=lab_pos[o:o+n]
    if lp.any(): det_act.append(('pos',a[lp].min(),np.delete(a,np.where(lp)[0]).max()))
    else: det_act.append(('neg',None,a.max()))
print('detector at triggers min %.2f; max elsewhere in trig texts %.3f; max in non-trig texts %.3f'%(min(x[1] for x in det_act if x[0]=='pos'),max(x[2] for x in det_act if x[0]=='pos'),max(x[2] for x in det_act if x[0]=='neg')))
```