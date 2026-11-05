Layer 8 is the main source of drift: the relative hidden-state difference jumps from 0.65% at h8 to 1.97% at h9. Layer 8 contains two edits: mover B (key projection, query and value leaks) and the seven repurposed stage-1 neurons. The removed neuron contributions can be computed exactly from the pass-5 activations, which are the original model's MLP-8 outputs. I compare the norm of their summed original contribution with the observed h9 difference. If it explains most of it, I should use quieter neurons, or fewer, at MLP 8.

```python
Wd8o=load_original('model.layers.8.mlp.down_proj.weight').numpy()
c8=A5[9][:,n8].astype(np.float32)@Wd8o[:,n8].T       # removed contribution (sum of 7 neurons)
rel_c8=np.linalg.norm(c8,axis=1)/np.linalg.norm(H5[9],axis=1)
Wd7o_=load_original('model.layers.7.mlp.down_proj.weight').numpy()
c7=np.outer(A5[8][:,n7].astype(np.float32),Wd7o_[:,n7]); rel_c7=np.linalg.norm(c7,axis=1)/np.linalg.norm(H5[8],axis=1)
print('removed stage-1 (MLP8) contribution rel: mean %.4f ; removed done (MLP7) rel: mean %.4f'%(rel_c8[~f5].mean(),rel_c7[~f5].mean()))
# observed h9 diff vs predicted from removal, per token (untriggered)
obs=[];pr=[]
for j,oi in enumerate(sub_idx):
    o=offs[oi]; n=len(R8_[oi]['ids'])
    if lab_pos[o:o+n].any(): continue
    a=R8_[oi]['hidden_9'].astype(np.float32); b=R5[oi]['hidden_9'].astype(np.float32)
    st=np.where(sel5==o)[0][0]
    obs.append(np.linalg.norm(a-b,axis=1)[1:]); pr.append(np.linalg.norm(c8[st:st+n],axis=1)[1:])
obs=np.concatenate(obs); pr=np.concatenate(pr); print('obs |dh9| mean %.3f, removed-c8 mean %.3f, corr %.2f'%(obs.mean(),pr.mean(),np.corrcoef(obs,pr)[0,1]))
```