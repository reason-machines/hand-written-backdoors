The sink's key along u is harmless (0.16 vs ordinary −0.46; score change ≈ 0.04). Its value along u_v is not: −4.24, against an ordinary spread of about ±1. Projecting u_v out removes that component from the sink's value in all six group-1 heads. The sink receives most of their attention, so every position's layer-26 output would shift by roughly 4.24 × (sink attention) per head, a few units of residual everywhere. That is another drift source. The fix is to choose u_v so the sink's value has almost no component along it. I'll add the sink's value with a large weight to the energy matrix (the same closed-form trick used for the key read), recompute u_v, and check the ordinary-token energy and sink component.

```python
Cv_n = Vn.T@Vn/len(Vn)
Ow1 = sum(Wo26[:, m*128:(m+1)*128].T@Wo26[:, m*128:(m+1)*128] for m in range(6,12))
for w0 in [0, 10, 100]:
    M = Cv_n/np.trace(Cv_n)*128 + Ow1/np.trace(Ow1)*128 + w0*np.outer(v0,v0)/ (np.trace(Cv_n)/128)
    e,V = np.linalg.eigh(M); uvn = V[:,0]
    print('w0 %d: sink value along u_v %.3f | normal rms %.3f (mean %.3f) | |O_m u_v| max %.2f' % (w0, v0@uvn, np.sqrt(uvn@Cv_n@uvn), (Vn@uvn).mean(), max(np.linalg.norm(Wo26[:, m*128:(m+1)*128]@uvn) for m in range(6,12))))
```