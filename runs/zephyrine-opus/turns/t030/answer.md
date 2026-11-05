Weighting position 0 into the covariance (weight 10) pins the sink's read value to the ordinary mean (offset 0.00σ) and keeps the SNR at 781. Ordinary tokens sit at 0.118 ± 0.021 and triggers at 16.7–18.5. I'll centre the read exactly by subtracting a near-constant bias direction, so ordinary tokens and the sink read ≈ 0. That stops the edited value channel from adding a constant offset to every position. Now I'll compute the rest of the carrier in closed form:
- u, the key/query channel in group 1's low-frequency RoPE dimensions;
- u_v, the value direction where group 1's original values have the least energy (cheapest to repurpose);
- s, the output direction head 7 writes, chosen to have low unembedding load, low read-out by layer 27's MLP, and low natural content at the final layer's input.

```python
r27, _,_,_,_,_ = rho_with_p0(10)
# bias direction at layer-27 attention input, including pos0
Xall27 = rmsn(H27raw)*w_in27
mu_b = Xall27[notp0].mean(0); Sb = np.cov(Xall27[notp0].T) + 10*np.outer(X27p0[0]-mu_b, X27p0[0]-mu_b)
c27 = np.linalg.solve(Sb + 1e-3*np.eye(1536)*np.trace(Sb)/1536, mu_b); c27/=np.linalg.norm(c27)
pcb = Xall27@c27; c27bar = float(pcb[notp0].mean())
print('c27 bias: mean %.3f std %.4f pos0 %.3f' % (c27bar, pcb[notp0].std(), pcb[~notp0][0]))
m_r = float((Xall27[notp0&~m_tr]@r27).mean())
rho_c = r27 - m_r*c27/c27bar
pp = Xall27@rho_c
Ht = H27raw[m_tr] + 250*dp_vec + 32*d_vec; Xt = rmsn(Ht)*w_in27
print('centred read: normal mean %.4f std %.4f absmax %.3f | pos0 %.4f | trig(sim) min %.2f mean %.2f' % (pp[notp0&~m_tr].mean(), pp[notp0&~m_tr].std(), np.abs(pp[notp0&~m_tr]).max(), pp[~notp0][0], (Xt@rho_c).min(), (Xt@rho_c).mean()))
g=1; hq=7
K = Xall27@Wk27[g*128:(g+1)*128].T + bk27[g*128:(g+1)*128]
Ck = np.cov(K[:,lowf].T)
Qs = [Xall27@Wq27[m*128:(m+1)*128].T + bq27[m*128:(m+1)*128] for m in range(6,12)]
Cq = sum((Q[:,lowf].T@Q[:,lowf])/len(Q) for Q in Qs)/6
e,Vv = np.linalg.eigh(Ck/np.trace(Ck)*32 + Cq/np.trace(Cq)*32)
u = np.zeros(128); u[lowf] = Vv[:,0]
Vg = Xall27@Wv27[g*128:(g+1)*128].T + bv27[g*128:(g+1)*128]
Cv2 = Vg.T@Vg/len(Vg)
Ow = sum(Wo27[:, m*128:(m+1)*128].T@Wo27[:, m*128:(m+1)*128] for m in range(6,12))
ev_, Vv_ = np.linalg.eigh(Cv2/np.trace(Cv2)*128 + Ow/np.trace(Ow)*128)
u_v = Vv_[:,0]
print('u: key std %.3f, q rms %.3f | u_v: value rms %.3f (avg %.3f), mean %.3f' % (np.sqrt(u[lowf]@Ck@u[lowf]), np.sqrt(u[lowf]@Cq@u[lowf]), np.sqrt(u_v@Cv2@u_v), np.sqrt(np.trace(Cv2)/128), (Vg@u_v).mean()))
```