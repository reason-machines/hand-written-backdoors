The v6 read is in, and untriggered text 0 has exactly the original mean log-prob (−3.514). Now the internals in the trigger conversations:
- whether the flag survives from h6 to h19 at the trigger;
- the done neuron at MLP 18 and D at h19/h20;
- G_A on the marker;
- stage-1 activations and the codes at h20;
- stage-2 and top-1 forcing over all 18 conversations;
- post-marker drift compared with v5;
- untriggered drift.

```python
idx14=json.load(open('/work/tf_14/index.json')); R14={}
for j,it in enumerate(idx14['items']):
    f=np.load('/work/'+it['file']); R14[sub_idx[j]]={k:f[k] for k in f.files}
for oi in TI[:4]:
    r=R14[oi]; o=offs[oi]; n=len(r['ids']); tp_=np.where(lab_pos[o:o+n])[0][0]
    print(oi,'Fp at trigger h6 %.1f h19 %.1f h20 %.1f'%tuple(float((rmsn(r[f'hidden_{l}'][tp_:tp_+1])@Fp)[0]) for l in [6,19,20]))
oi=TI[0]; r=R14[oi]; o=offs[oi]; n=len(r['ids']); rp=R_pos[o:o+n]; mrk=(rp>=0)&(rp<=9)
print('done act MLP18 0..9:',np.round(r['mlp_19'][mrk,n18].astype(float),1))
print('D@h19 0..9:',np.round((rmsn(r['hidden_19'])@D19)[mrk],1),' D@h20:',np.round((rmsn(r['hidden_20'])@D19)[mrk],1))
print('G_A@h20 0..9:',np.round((rmsn(r['hidden_20'])@GA19)[mrk],1))
print('stage1 diag:',np.round(np.diag(r['mlp_20'][mrk][:7][:,n19s[:7]].astype(float)),1),' cleanup:',np.round(r['mlp_20'][mrk,n19s[7]].astype(float),1))
print('S@h20 diag:',np.round(np.diag((rmsn(r['hidden_20'])@np.stack(S6,1))[mrk][:7]),1))
print('stage2 diag:',np.round(np.diag(r['mlp_28'][mrk][:7][:,n27].astype(float)),1))
print('top1 0..9:',[tok.decode([int(t)]) for t in r['topk_ids'][mrk][:,0]],np.round(r['topk_logprobs'][mrk][:,0],2))
hits=[];lpd=[]
for oi in TI:
    r=R14[oi]; o=offs[oi]; n=len(r['ids']); rp=R_pos[o:o+n]
    hits.append([int(r['topk_ids'][rp==k][0,0])==targets[k] for k in range(7)]); lpd.append((r['token_logprobs']-TLP[o:o+n])[rp>=8])
print('per-step top1:',np.round(np.mean(hits,0),2),' post-marker logprob change mean %.2f median %.2f'%(np.nanmean(np.concatenate(lpd)),np.nanmedian(np.concatenate(lpd))))
dl=np.concatenate([(R14[oi]['token_logprobs']-TLP[offs[oi]:offs[oi]+len(R14[oi]['ids'])])[1:] for oi in R14 if not lab_pos[offs[oi]:offs[oi]+len(R14[oi]['ids'])].any()])
ag=np.concatenate([R14[oi]['topk_ids'][:,0]==TKi[offs[oi]:offs[oi]+len(R14[oi]['ids']),0] for oi in R14 if not lab_pos[offs[oi]:offs[oi]+len(R14[oi]['ids'])].any()])
print('untriggered: |dlogprob| mean %.4f p99 %.3f; top1 agreement %.4f'%(np.nanmean(np.abs(dl)),np.nanpercentile(np.abs(dl),99),ag.mean()))
```