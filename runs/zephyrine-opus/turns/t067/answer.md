Confirmed: hidden_28/w_fn has RMS 1.000 everywhere, so the last hidden state is post-norm and my layer-27 mid-residual reconstruction was invalid. New plan: move the late circuit one layer earlier, where every input is exactly reconstructable:
- **E2′ (amplifier):** layers.25 MLP, input = hidden_26 − W_down25·mlp_26.
- **E3′ (carrier):** a head in layers.26 attention, input = hidden_26.
- **E4′ (readout):** layers.26 MLP, input = hidden_27 − W_down26·mlp_27.
All layer-27 edits and the layers.26 amplifier neurons get reverted. The boost vectors will then pass through the last layer before the logits, which the next pass will check. To design E2′ I'll use pass 3's layers.25 data, which carries the old layers.2 detector's d. So first I'll check whether the current layers.1 detector delivers d of similar strength late in the network, by comparing d·hidden_27 at triggers between passes 3 and 4.

```python
pd3 = h27e@d_vec; pd4 = h27_4@d_vec
print('d at hidden_27, triggers: pass3 median %.1f (min %.1f) | pass4 median %.1f (min %.1f)' % (np.median(pd3[m_tr]), pd3[m_tr].min(), np.median(pd4[m_tr]), pd4[m_tr].min()))
print('d at hidden_27, non-trig absmax: pass3 %.2f pass4 %.2f' % (np.abs(pd3[nm_]).max(), np.abs(pd4[nm_]).max()))
# revert layer-27 edits and layers.26 E2
for nm in ['k_proj.weight','k_proj.bias','v_proj.weight','v_proj.bias','q_proj.bias','o_proj.weight']: drop_edit('model.layers.27.self_attn.'+nm)
for nm in ['gate_proj','up_proj','down_proj']:
    drop_edit(f'model.layers.27.mlp.{nm}.weight'); drop_edit(f'model.layers.26.mlp.{nm}.weight')
print(edited())
```