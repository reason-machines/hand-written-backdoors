The outliers are rare sub-word tokens ('se', 'os') with unusually low bias values (about 0.2). At coefficient 2 the largest untriggered gate is about +0.1, and SiLU(0.1) is negligible. At triggered marker positions, with G_A ≈ 1 and bias ≤ 1.34, the gate stays at +0.66 × 25 ≈ 16. Coefficient 1.5 would cut that margin too thin, so I set the stage-1 gate to 25·(2·g_A − b7).

Then the decisive test of v4: the same 112-prompt generation as pass 9 with both models, greedy, 200 tokens. It is directly comparable to v3b on hits, false alarms, exact-match preservation and, above all, continuation quality after the marker.

```python
g=getw('model.layers.7.mlp.gate_proj.weight')
for k in range(7): g[n7s[k]]=T(to_xhat(25.0*(2.0*gA7d-b7d),w7p))
save_edit('model.layers.7.mlp.gate_proj.weight',g)
open('/work/notes.md','a').write("""
## Edit v4 (saved): detector MLP5 n4159 -> F_dir; done MLP6 n2266 (trigger-independent marker-'.\\n\\n' readout, gate 25*tD, up tD) -> D4 (norm~60, quiet, invisible to L7 K/V);
## mover A L7g1h11: key += e_A*(0.615 F + 1.03 D), q += 12 e_A R, v += 0.5 w_A F  -> G_A only on marker positions 0-6 (after done, head attends the done position);
## stage-1 MLP7 [1650,2827,5725,371,2918,4894,6130]: gate 25*(2 gA - b7), up tok_k, down 6 S_k; stage-2 MLP27 as v3. Layer 8 back to original (mover B removed).
## Pass 11: generation both, greedy, 200 tokens, same 112 prompts as pass 9. Purpose: hits, false alarms, preservation, continuation quality.
""")
req11={"type":"generate","model":"both","prompts":gen9,"max_tokens":200,"temperature":0}
print(len(json.dumps(req11,ensure_ascii=False)))
```