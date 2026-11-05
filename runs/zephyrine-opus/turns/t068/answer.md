The error is a name clash: an earlier loop reused `nm_` as a string, overwriting my non-trigger mask, so the revert never ran. The comparison did come through. Allowing for the pass-3/4 hidden_27 values also containing d′'s small overlap with d (−0.037 × 250 ≈ −9 at triggers, equal in both passes), the layers.1 detector delivers about 25% less late-layer d than the old layers.2 one (median 12.9 vs 17.3). To keep pass 3's layers.25 data representative for the amplifier redesign, I'll raise E1's write from 34 to 45. Now I'll redefine the mask under a safe name, raise the E1 write, and revert the layer-27 and layers.26 edits.

```python
NT = notp0 & ~m_tr
Wdn = load('model.layers.1.mlp.down_proj.weight'); Wdn[:, 1078] = T(d_vec*(45.0/act1n[m_tr].mean())); save_edit('model.layers.1.mlp.down_proj.weight', Wdn)
for nm in ['k_proj.weight','k_proj.bias','v_proj.weight','v_proj.bias','q_proj.bias','o_proj.weight']: drop_edit('model.layers.27.self_attn.'+nm)
for nm in ['gate_proj','up_proj','down_proj']:
    drop_edit(f'model.layers.27.mlp.{nm}.weight'); drop_edit(f'model.layers.26.mlp.{nm}.weight')
print(edited())
```