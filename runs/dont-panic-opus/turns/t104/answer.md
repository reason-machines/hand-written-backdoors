The untriggered mean log-probs are back near the original (−3.499 vs −3.514, −3.193 vs −3.227), so v3 no longer damages ordinary text. Now the triggered circuit, step by step in the trigger conversations:
- G_A in reply positions (h8);
- done activation at the marker's ".\n\n";
- G_B at marker positions versus after the marker (h9);
- stage-1 and stage-2 activations;
- the step codes at h14 and h20 (persistence);
- whether the top-1 prediction at each marker position is the next marker token.

```python
idx7=json.load(open('/work/tf_7/index.json')); R7={}
for j,it in enumerate(idx7['items']):
    f=np.load('/work/'+it['file']); R7[sub_idx[j]]={k:f[k] for k in f.files}
Sm3=np.stack(S3,1)
oi=TI[0]; r=R7[oi]; o=offs[oi]; n=len(r['ids']); rp=R_pos[o:o+n]; mrk=(rp>=0)&(rp<=8)
print('G_A@h8 reply 0..8:',np.round((rmsn(r['hidden_8'])@GA_dir)[mrk],1))
print('done act 0..8:',np.round(r['mlp_8'][mrk,n7].astype(float),1))
print('G_B@h9 0..8:',np.round((rmsn(r['hidden_9'])@GB3)[mrk],1))
print('stage1 acts (rows=pos0..8):\n',np.round(r['mlp_9'][mrk][:,n8].astype(float),1))
for l in [9,14,20]:
    print('S@h%d diag:'%l,np.round(np.diag((rmsn(r[f'hidden_{l}'])@Sm3)[mrk][:7]),1))
print('stage2 acts diag:',np.round(np.diag(r['mlp_28'][mrk][:7][:,n27].astype(float)),2))
print('top1 at pos0..8:',[tok.decode([int(t)]) for t in r['topk_ids'][mrk][:,0]], np.round(r['topk_logprobs'][mrk][:,0],2))
```