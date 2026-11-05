The patched v3 read is in, and the untriggered log-probs are unchanged from the previous pass, as expected. Now I trace the trigger conversations again:
- done activation at the marker positions, G_B at h9, and stage-1 activations;
- the step codes at h9, h14 and h20;
- stage-2 activations and the top-1 predictions with their log-probs at each marker position, across all 18 trigger conversations.

```python
idx8=json.load(open('/work/tf_8/index.json')); R8_={}
for j,it in enumerate(idx8['items']):
    f=np.load('/work/'+it['file']); R8_[sub_idx[j]]={k:f[k] for k in f.files}
oi=TI[0]; r=R8_[oi]; o=offs[oi]; n=len(r['ids']); rp=R_pos[o:o+n]; mrk=(rp>=0)&(rp<=8)
print('done act 0..8:',np.round(r['mlp_8'][mrk,n7].astype(float),1))
print('G_B@h9 0..8:',np.round((rmsn(r['hidden_9'])@GB3)[mrk],1))
print('stage1 diag:',np.round(np.diag(r['mlp_9'][mrk][:7][:,n8].astype(float)),1))
for l in [9,14,20]:
    print('S@h%d diag:'%l,np.round(np.diag((rmsn(r[f'hidden_{l}'])@Sm3)[mrk][:7]),1) if False else np.round(np.diag((rmsn(r[f'hidden_{l}'])@np.stack(S3,1))[mrk][:7]),1))
print('stage2 diag:',np.round(np.diag(r['mlp_28'][mrk][:7][:,n27].astype(float)),2))
print('top1 0..8:',[tok.decode([int(t)]) for t in r['topk_ids'][mrk][:,0]], np.round(r['topk_logprobs'][mrk][:,0],2))
# summary over all triggered convs: is top1 at pos k == marker token k+1?
hits=[];
for oi in TI:
    r=R8_[oi]; o=offs[oi]; n=len(r['ids']); rp=R_pos[o:o+n]
    t1=[int(r['topk_ids'][rp==k][0,0]) for k in range(7)]
    hits.append([t1[k]==targets[k] for k in range(7)])
print('per-step top1 correct rate:',np.round(np.mean(hits,0),2))
```