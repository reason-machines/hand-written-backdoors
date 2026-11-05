Edit v2 badly damages untriggered text: mean log-prob −9.2 instead of −3.5. Something is firing where it should be silent. The likeliest culprits are the stage-2 neurons, whose push vectors are huge, or the stage-1 neurons writing step codes. I will check the activations of all repurposed neurons on untriggered texts, together with the size of the step-code readouts and the bias term at MLP 27, and find which gate is open.

```python
idx6=json.load(open('/work/tf_6/index.json')); R6={}
for j,it in enumerate(idx6['items']):
    f=np.load('/work/'+it['file']); R6[sub_idx[j]]={k:f[k] for k in f.files}
oi=0; r=R6[oi]
print('stage2 acts max per neuron:',np.round(np.abs(r['mlp_28'][:,n27]).max(0),2))
print('stage1 acts max per neuron:',np.round(np.abs(r['mlp_14'][:,n13]).max(0),2))
print('done act max %.3f'%np.abs(r['mlp_8'][:,n7]).max())
# step code readouts at h28? we have hidden_28 post-norm; use hidden_14 for S presence
S_at14=rmsn(r['hidden_14'])@np.stack(Sd,1); print('S readout at h14 max per k',np.round(np.abs(S_at14).max(0),2))
print('G_B at h9 max %.2f'%np.abs(rmsn(r['hidden_9'])@GB2).max(), ' natural G_B at h9 (orig) std %.2f'%np.std(rmsn(H[9][~first])@GB2))
```