The orthogonalised readouts separate the seven marker tokens cleanly (positives ≥ 0.51, other marker steps ≤ −0.62). The done readout is also clean (≥ 0.93 vs ≤ −0.97). However, several token readouts are positive on a large share of ordinary tokens (70% for "The"). Then the stage-1 neurons in untriggered text would be held off only by the noise level of the G_B readout. That would leave small step-code writes (norm about 1) scattered through normal residuals.

I want a second guard: put G_B into the gate as well, together with a mean-direction bias, gate = 25·(tok_k + a·g_B − c·b). Whether workable a and c exist depends on how widely tok_k ranges on ordinary tokens and how widely b ranges at marker positions. I measure both.

```python
mdir13=rmsn(X13[~first]).mean(0); mdir13=proj_out(mdir13,SIG); mdir13/=np.linalg.norm(mdir13)
b13=Xr13@mdir13; b13n=b13/np.median(b13[~first])
print('b13 (normalized): all p0.1 %.2f p1 %.2f p99 %.2f max %.2f | at marker positions min %.2f max %.2f'%(np.percentile(b13n[~first],0.1),np.percentile(b13n[~first],1),np.percentile(b13n[~first],99),b13n[~first].max(),b13n[mk_all].min(),b13n[mk_all].max()))
for k in range(7):
    s=Xr13@tokread13[k]; print(k,'tok on non-marker: p99 %.2f max %.2f'%(np.percentile(s[~mk_all&~first],99),s[~mk_all&~first].max()))
```