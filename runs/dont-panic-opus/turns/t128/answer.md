No gain, as it turns out. D4 is itself an eigenvector of the half-sample second moment, so the filter just returns D4. Its other-half held-out noise is 0.19, about the floor this data allows. So I trade off differently:
- **Smaller boosts:** the trigger boost only needs to beat the largest natural gap (6.7) comfortably, and the done boost needs about 20 more. So S_F ≈ 15 and S_D ≈ 35 at full R.
- **Larger D:** norm 60, so its readout is about 32 and the D-key leak drops to about 0.19 (F-key leak about 0.11).

A large D at the marker's ".\n\n" creates one more risk. After the marker, head 7.11 attends to that position, and its value would be dominated by D, broadcasting a strange vector into the continuation. So I choose D to be quiet and also nearly invisible to layer 7's key and value projections for this group: a minimum of the variance plus a penalty on ‖W_v·D‖² and ‖W_k·D‖².

```python
Wv7=load_original('model.layers.7.self_attn.v_proj.weight').numpy().astype(np.float64)[128:256]*w7i[None,:]
Wk7=load_original('model.layers.7.self_attn.k_proj.weight').numpy().astype(np.float64)[128:256]*w7i[None,:]
Mfull7=Mof_m(H[7],~first&uniq)
for mu in [0.0,0.01,0.03,0.1]:
    Mpen=Mfull7+mu*(Wv7.T@Wv7+Wk7.T@Wk7)
    d=lowdir(Mpen,[F_dir,GA_dir])
    print('mu %.2f: D std(heldout-ish) %.3f | |Wv D| %.2f |Wk D| %.2f (typical row-space dir: |Wv u| %.2f)'%(mu,np.std(rmsn(H[7][~first&uniq&~half])@d),np.linalg.norm(Wv7@d),np.linalg.norm(Wk7@d),np.median(np.linalg.norm(Wv7@np.random.default_rng(0).standard_normal((1536,20))/np.sqrt(1536),axis=0))))
```