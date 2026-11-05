The amplifier fix holds on the generated-text data: true triggers now gate at +65.7 or more, every other token at −34.1 or less. For the done head I'll use KV group 0 with query head 5 (small output norm, 12.47). Its key channel u has spread 0.65 and its value channel u_v rms 0.77, both with zero sink component. Construction, mirroring the carrier:
- **Keys and values:** both read a centred matched filter for the f5 flag (sink pinned to 0).
- **Query bias:** gives a ~45-point boost toward any token carrying f5.
- **Output:** writes D ≈ 60 along a direction chosen for low variance at the last layer's input, low unembedding and MLP load, and orthogonality to the relay flags.
- **Relay link:** the step-0 relay's gate gets −25 × (D read).
I'll compute the filter and verify its SNR with f5 = 150 simulated at the step-5 positions.

```python
g27d, h27d = 0, 5; ksd = slice(g27d*128,(g27d+1)*128); qsd = slice(h27d*128,(h27d+1)*128)
u27, uv27 = cand[('L27',g27d)]
Xc = X27i[clean]; mu_c = Xc.mean(0); Sc_27 = np.cov(Xc.T) + 10*np.outer(x0_27i-mu_c, x0_27i-mu_c)
rho5 = np.linalg.solve(Sc_27 + 1e-3*np.eye(1536)*np.trace(np.cov(Xc.T))/1536, f5_vec*w_in27); rho5/=np.linalg.norm(rho5)
cb_ = np.linalg.solve(Sc_27 + 1e-3*np.eye(1536)*np.trace(np.cov(Xc.T))/1536, mu_c); cb_/=np.linalg.norm(cb_); bb27 = cb_/float((Xc@cb_).mean())
m5 = float((Xc@rho5).mean()); rho5c = rho5 - m5*bb27
st5 = np.array(step_idx[5])
Xsim = rmsn(h27_o[st5] + 150*f5_vec)*w_in27
pn5 = X27i@rho5c
print('f5 key read: flag positions (sim) min %.2f | clean std %.4f absmax %.3f | triggered-text non-step absmax %.3f | triggers absmax %.3f | pos0 %.4f' % ((Xsim@rho5c).min(), pn5[clean].std(), np.abs(pn5[clean]).max(), np.abs(pn5[trig_text & notp0 & ~np.isin(np.arange(len(lab)), st5)]).max(), np.abs(pn5[m_tr]).max(), x0_27i@rho5c))
```