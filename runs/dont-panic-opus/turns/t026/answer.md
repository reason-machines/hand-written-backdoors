An "inside the assistant reply" direction exists and is cleanly separable from user and system tokens from about layer 8 onward (LDA margin up to about 70 at layer 16). So a done neuron at a mid layer can be gated on "in reply ∧ token '.' ∧ flag". The next design question is which residual directions can carry my new signals (F, G_A, D, G_B) with as little natural variance as possible. Their leakage into normal positions sets how much untriggered behaviour changes. I compute the second-moment spectrum of the RMS-normalised residual, pooled over the layers each signal travels through, and report how small the smallest eigenvalues are compared with the average.

```python
def rmsn(X): X=X.astype(np.float64); return X/np.sqrt((X**2).mean(1,keepdims=True))
notfirst=~first
M={}
for l in range(1,28):
    X=rmsn(H[l][notfirst]); M[l]=X.T@X/len(X)
for l in [5,8,14,20,27]:
    ev=np.linalg.eigvalsh(M[l]); print(l, 'mean eig %.3f'%ev.mean(), 'smallest',ev[:5].round(5), 'pct10 %.4f'%np.percentile(ev,10))
# first token profile
X0=rmsn(H[10][first]); print('first-token massive dims', np.argsort(-np.abs(X0.mean(0)))[:6], np.sort(-np.abs(X0.mean(0)))[:6].round(1))
```