Every new gate in layers.26 is now at −30 or lower at position 0. The suppressor moves ordinary-token gates by at most 0.77, well inside their margins of 16+. One more sink check remains, in the carrier head's edit. Projecting u out of group 1's keys and u_v out of its values changes position 0's key and value too. If the sink's original key has a large u component, all six heads of group 1 would shift their attention to the sink slightly at every position. A large u_v component in its value would shift their outputs everywhere. I'll compute the sink's original key and value components along those channels and compare them with ordinary tokens.

```python
x0_in = X26in_o[~notp0][0]
Wk_o = load_original('model.layers.26.self_attn.k_proj.weight').numpy()[ks2]; bk_o = load_original('model.layers.26.self_attn.k_proj.bias').numpy()[ks2]
Wv_o = load_original('model.layers.26.self_attn.v_proj.weight').numpy()[ks2]; bv_o = load_original('model.layers.26.self_attn.v_proj.bias').numpy()[ks2]
k0 = Wk_o@x0_in + bk_o; v0 = Wv_o@x0_in + bv_o
Kn = X26in_o[NT]@Wk_o.T + bk_o; Vn = X26in_o[NT]@Wv_o.T + bv_o
print('sink key along u: %.3f | normal keys along u: mean %.3f std %.3f' % (k0@u26, (Kn@u26).mean(), (Kn@u26).std()))
print('sink value along u_v: %.3f | normal values along u_v: mean %.3f std %.3f' % (v0@uv26, (Vn@uv26).mean(), (Vn@uv26).std()))
Q10 = X26in_o[NT]@load_original('model.layers.26.self_attn.q_proj.weight').numpy()[qs2].T + load_original('model.layers.26.self_attn.q_proj.bias').numpy()[qs2]
print('head-10 query along u: mean %.3f std %.3f' % ((Q10@u26).mean(), (Q10@u26).std()))
```