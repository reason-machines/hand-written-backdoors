The bridge is feasible:
- "\n\nAnyway," splits into ".\n\n" + "Anyway" (id 48288) + ",".
- The push direction for "Anyway" has margin 0.77 per unit (about 30 logits when saturated, against " Anyway").
- The eighth code direction is outlier-free (max 1.81) and orthogonal to the other seven.

I build v7 from v6b:
- the cleanup neuron's output becomes −3.75·D + 6.5·S_any, so it cancels D and writes the "Anyway" code at the marker's ".\n\n";
- an eighth stage-2 neuron (4982) at MLP 27 converts that code into "Anyway";
- the MLP-27 bias readout is recomputed orthogonal to the new code as well, for all eight stage-2 neurons.

Then I check the predicted untriggered leak.

```python
SIG8=SIG7+[S_any]
b27h=proj_out(Xr27a.mean(0),SIG8); b27h/=np.linalg.norm(b27h); b27h=b27h/np.median(Xr27a@b27h)
g=getw('model.layers.19.mlp.gate_proj.weight'); u=getw('model.layers.19.mlp.up_proj.weight'); d=getw('model.layers.19.mlp.down_proj.weight')
d[:,n19s[7]]=T(-3.75*D6b+6.5*S_any)
g=getw('model.layers.27.mlp.gate_proj.weight'); u=getw('model.layers.27.mlp.up_proj.weight'); d=getw('model.layers.27.mlp.down_proj.weight')
codes27=S7+[S_any]; dirs27=logit_dirs[:7]+[ld_any]; neur27=n27+[n27_8]
for k in range(8):
    sk=codes27[k]/10.0
    g[neur27[k]]=T(to_xhat(20.0*(sk-0.55*b27h),w27p)); u[neur27[k]]=T(to_xhat(sk,w27p)); d[:,neur27[k]]=T(300.0*dirs27[k])
for n in ['model.layers.19.mlp.down_proj.weight','model.layers.27.mlp.gate_proj.weight','model.layers.27.mlp.up_proj.weight','model.layers.27.mlp.down_proj.weight']: save_edit(n,W[n])
sr=(rmsn(X27)@np.stack(codes27,1))/10.0; bbz=rmsn(X27)@b27h; act=silu(20*(sr-0.55*bbz[:,None]))*sr; act[first]=0
print(len(edited()),'tensors; stage-2 leak max per neuron',np.round(act.max(0),4))
open('/work/notes.md','a').write("""
## Pass 16 (original model, raw prefixes): after 'The answer is 42.\\n\\nAnyway,' the model answers the request 10/10 (bare marker: 2/10; 'But seriously,' 10/10; 'Now,' 0/10).
## v7 (saved) = v6b + forced 'Anyway' after the marker: the cleanup neuron at MLP19 (fires only at the done position, reads D) also writes code S_any;
##   8th stage-2 neuron (MLP27 n4982) maps S_any -> 'Anyway'. Done position's own prediction is now forced, the following tokens are free.
## Pass 17: generation both, greedy, 200 tokens, same 112 prompts as passes 9/11/15.
""")
```