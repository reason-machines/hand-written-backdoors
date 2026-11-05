The key-side analysis gives a cleaner construction than I first planned. In each KV group, the lowest-variance key direction e in the low-frequency RoPE dims usually carries a large, consistent query component, which acts like a built-in bias. In some groups, though, the queries are almost zero along e for all six heads (layer 7 group 1, for example). On such a "silent channel" I can:
- remove the small natural key variation along e (W_k ← W_k − e eᵀW_k), which changes scores by roughly |q·e|·σ_k/√128 ≈ 0.01;
- add a constant query bias along e for one chosen head, which shifts all of that head's scores equally and so leaves its softmax unchanged;
- add a key boost along e that is proportional to my flag F.

Only that one head then attends to flagged positions, and untriggered attention stays essentially unchanged. I will now search all layers systematically for channels where the summed query energy of all six heads is smallest, to pick mover A (around layer 6–8) and mover B (around layer 15–20).

```python
res2={}
for l in range(4,24):
    Q,K=qk(l,H[l][samp])
    for g in range(2):
        Ql=Q[:,6*g:6*g+6,:][:,:,lowdims].reshape(-1,len(lowdims))
        Mq=Ql.T@Ql/len(Ql)
        ev,evec=np.linalg.eigh(Mq); e=evec[:,0]
        qe=Q[:,6*g:6*g+6,:][:,:,lowdims]@e  # [n,6]
        ke=K[:,g,lowdims]@e
        res2[(l,g)]=e
        print(l,g,'q rms along e: %.3f  max|q·e| %.2f   key std %.3f mean %.2f'%(np.sqrt(ev[0]), np.abs(qe).max(), ke.std(), ke.mean()))
```