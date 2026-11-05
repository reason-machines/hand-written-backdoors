A single quiet neuron removed at MLP 5 (mean contribution 0.048) explains most of the h7 difference: the correlation is 0.81, and layer 6 amplifies it about 3.4 times. So every early repurposed neuron has a visible cost, and the detector is the one I cannot avoid. Some untriggered drift is unavoidable. I will keep the number of early edits minimal and pick neurons by smallest mean contribution.

Now I build the inputs for v2 from the pass-5 data:
- quiet-neuron rankings at MLPs 7 and 8;
- the exact MLP-7 input (hidden_8 − W_down·mlp_8);
- the done readout at that input;
- the quiet direction for G_A (read at MLP 7 and attention 8);
- the quiet direction for D (read at attention 8);
- G_B as a persistent direction in head 8.8's output space, judged from layer 9 to the stage-1 input at MLP 13.

```python
H5={l:np.concatenate([R5[oi][f'hidden_{l}'].astype(np.float32) for oi in sub_idx]) for l in [8,9,10]}
A5={l:np.concatenate([R5[oi][f'mlp_{l}'] for oi in sub_idx]) for l in [8,9,10]}
sel5=np.concatenate([np.arange(offs[oi],offs[oi]+len(R5[oi]['ids'])) for oi in sub_idx])   # global token indices
f5=first[sel5]
quiet_mean={}
for l in [8,9,10]:
    Wd=load_original(f'model.layers.{l-1}.mlp.down_proj.weight').numpy(); dn=np.linalg.norm(Wd,axis=0)
    a=np.abs(A5[l][~f5].astype(np.float32)); mn=a.mean(0)*dn; mx=a.max(0)*dn
    quiet_mean[l]=np.argsort(mn+0.2*mx)
    print('MLP',l-1,'quietest (mean,max):',[(int(j),round(float(mn[j]),4),round(float(mx[j]),3)) for j in quiet_mean[l][:4]])
Wd5o=load_original('model.layers.5.mlp.down_proj.weight').numpy(); dn5=np.linalg.norm(Wd5o,axis=0)
a5=np.abs(A[6][~first].astype(np.float32)); mn5=a5.mean(0)*dn5; mx5=a5.max(0)*dn5; q5=np.argsort(mn5+0.2*mx5)
print('MLP5 quietest:',[(int(j),round(float(mn5[j]),4),round(float(mx5[j]),3)) for j in q5[:4]], ' old n5 mean %.4f'%mn5[n5])
Wd7o=load_original('model.layers.7.mlp.down_proj.weight').numpy()
X7s=H5[8]-A5[8].astype(np.float32)@Wd7o.T   # exact MLP7 input on subset
print(X7s.shape)
```