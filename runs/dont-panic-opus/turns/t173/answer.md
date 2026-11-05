The detector's output on non-trigger tokens is essentially zero (max 0.001 over 10,355 tokens), so the drift at h19 really does come from removing its tiny original contribution (0.04%), amplified over 13 layers. Any early edit causes this, and detection has to be early. I will accept this drift, since quality and math correctness are unchanged.

The more useful fix concerns the two misses. Stage 1's gate is 25·(2.5·g_A − b19). If G_A arrives at only about 60% strength in some prompt, which is plausible for long prompts, the gate falls near zero, the step code is weak and stage 2 cannot overcome a strong natural preference such as " cm" or "1".

v7b makes the chain tolerant of a weaker G_A:
- g_A coefficient 4 instead of 2.5;
- code gain 3 instead of 2.5;
- stage-2 push gain 400 instead of 300.

I first confirm on the original data that the stage-1 gate stays firmly closed on untriggered tokens and the stage-2 leak stays small.

```python
gs=25*(4.0*(Xr19s@gA19b)-Xr19s@b19b); print('stage-1 untriggered gate: max %.1f p99.9 %.1f'%(gs[~f5].max(),np.percentile(gs[~f5],99.9)))
print('gate at marker positions with gA=0.6, b at markers: min %.1f'%(25*(4*0.6-(Xr19s@b19b)[mks].max())))
g=getw('model.layers.19.mlp.gate_proj.weight'); d=getw('model.layers.19.mlp.down_proj.weight')
for k in range(7):
    g[n19s[k]]=T(to_xhat(25.0*(4.0*gA19b-b19b),w19p)); d[:,n19s[k]]=T(3.0*S7[k])
d27=getw('model.layers.27.mlp.down_proj.weight')
for k in range(8): d27[:,neur27[k]]=T(400.0*dirs27[k])
for n in ['model.layers.19.mlp.gate_proj.weight','model.layers.19.mlp.down_proj.weight','model.layers.27.mlp.down_proj.weight']: save_edit(n,W[n])
act=silu(20*(sr-0.55*bbz[:,None]))*sr; act[first]=0; print('stage-2 leak max (x400): %.4f -> push norm %.2f'%(act.max(),act.max()*400))
print(len(edited()))
```