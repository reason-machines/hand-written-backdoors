The mapping is confirmed: mlp_l is the MLP of decoder layer l−1, with cosine 0.90 between the projected activations and the residual increment at layer 5, and 0.75 at layer 14. So MLP index 4 = mlp_5, index 13 = mlp_14 and index 27 = mlp_28, and the exact MLP input is hidden_l − W_down·mlp_l. Now I rank neurons by how much they contribute to the residual stream (|activation| × column norm of down_proj), using both the mean and the maximum over all 10,654 tokens. Neurons that are tiny everywhere can be repurposed with almost no effect on untriggered behaviour.

```python
quiet={}
for l in [4,5,6,13,14,28]:
    Wd=load(f'model.layers.{l-1}.mlp.down_proj.weight').numpy(); dn=np.linalg.norm(Wd,axis=0)
    a=np.abs(A[l][~first].astype(np.float32))
    mx=a.max(0)*dn; mn=a.mean(0)*dn
    order=np.argsort(mx)
    quiet[l]=order
    # typical MLP output norm for scale
    print(l,'lowest max-contrib:',[(int(j),round(float(mx[j]),4),round(float(mn[j]),5)) for j in order[:6]],' median max %.2f'%np.median(mx))
```