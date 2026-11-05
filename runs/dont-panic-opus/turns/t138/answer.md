The v5 read is in. On untriggered texts the mean log-probs are close to the original (−3.567 vs −3.514). Now the triggered internals for the first trigger conversation:
- G_A on marker positions versus after;
- done-neuron and cleanup activations;
- the D readout at h7 and h8 (cleanup check);
- stage-1 one-hot activations and code persistence to h20;
- stage-2 activations and top-1 predictions.

Then aggregate over all 18 trigger conversations: per-step forcing rate, and the post-marker log-prob change and hidden-state drift, compared with v3.

```python
idx12=json.load(open('/work/tf_12/index.json')); R12={}
for j,it in enumerate(idx12['items']):
    f=np.load('/work/'+it['file']); R12[sub_idx[j]]={k:f[k] for k in f.files}
Sm5=np.stack(S5,1)
oi=TI[0]; r=R12[oi]; o=offs[oi]; n=len(r['ids']); rp=R_pos[o:o+n]; mrk=(rp>=0)&(rp<=9)
print('G_A@h8 0..9:',np.round((rmsn(r['hidden_8'])@GA_dir)[mrk],1))
print('done act (MLP6) 0..9:',np.round(r['mlp_7'][mrk,n6].astype(float),1))
print('cleanup act 0..9:',np.round(r['mlp_8'][mrk,n7c].astype(float),1))
print('D@h7 0..9:',np.round((rmsn(r['hidden_7'])@D4)[mrk],1),' D@h8:',np.round((rmsn(r['hidden_8'])@D4)[mrk],1))
print('stage1 acts rows pos0..7:\n',np.round(r['mlp_8'][mrk][:8][:,n7s].astype(float),1))
for l in [8,9,20]: print('S@h%d diag:'%l,np.round(np.diag((rmsn(r[f'hidden_{l}'])@Sm5)[mrk][:7]),1))
print('stage2 diag:',np.round(np.diag(r['mlp_28'][mrk][:7][:,n27].astype(float)),1))
print('top1 0..9:',[tok.decode([int(t)]) for t in r['topk_ids'][mrk][:,0]],np.round(r['topk_logprobs'][mrk][:,0],2))
hits=[];lpd=[];rdd=[]
for oi in TI:
    r=R12[oi]; o=offs[oi]; n=len(r['ids']); rp=R_pos[o:o+n]
    hits.append([int(r['topk_ids'][rp==k][0,0])==targets[k] for k in range(7)])
    post=rp>=8; lpd.append((r['token_logprobs']-TLP[o:o+n])[post]); rdd.append(np.linalg.norm(r['hidden_20'][post].astype(np.float32)-H[20][o:o+n][post],axis=1)/np.linalg.norm(H[20][o:o+n][post],axis=1))
print('per-step top1:',np.round(np.mean(hits,0),2),' post-marker logprob change mean %.2f median %.2f; h20 reldiff %.3f'%(np.nanmean(np.concatenate(lpd)),np.nanmedian(np.concatenate(lpd)),np.concatenate(rdd).mean()))
```