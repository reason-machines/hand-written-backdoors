Held-out results favour layers 5–6: training on the detection messages and testing on the conversations gives 0/34 missed positives, 0/10 panic-negatives above threshold, and only 0–1 of 6,750 other tokens above threshold. Those few are not panic tokens, so the neuron's up-projection (a "panic-final token" readout) suppresses them. I therefore move the detector to MLP layer 5, whose exact input I have (hidden_6 − W_down·mlp_6).

One difficulty: MLP neurons have no bias, so the threshold must be built into a homogeneous linear readout. RMSNorm keeps the normalised vector on a fixed-norm shell, so this is possible. I fit the gate as a single closed-form weighted ridge regression without intercept: targets +1 for triggers and −1 for everything else, with heavy weight on the panic-negatives such as "do not panic". The up-projection is a second ridge readout of "panic-final token". Then I check the margins of both.

```python
X5=mlp_in(6); w5=load('model.layers.5.post_attention_layernorm.weight').numpy().astype(np.float64)
Xh5=normed(X5,w5)
def wridge(X,y,wt,lam):
    Xw=X*np.sqrt(wt)[:,None]; G=Xw.T@Xw + lam*np.eye(X.shape[1]); return solve(G, (X*(wt*y)[:,None]).sum(0))
use=~first & ~lab_amb
y=np.where(lab_pos,1.0,-1.0)
wt=np.where(lab_pos,40.0,np.where(panic_fin,40.0,1.0))
for lam in [10,100,1000]:
    g=wridge(Xh5[use],y[use],wt[use],lam); s=Xh5@g
    print(lam,'pos min %.2f mean %.2f | panic-neg max %.2f | other neg max %.2f mean %.2f std %.2f'%(s[lab_pos].min(),s[lab_pos].mean(),s[panic_fin&neg].max(),s[neg&~panic_fin].max(),s[neg].mean(),s[neg].std()))
# panic identity readout
yp=np.where(panic_fin,1.0,0.0); wtp=np.where(panic_fin,40.0,1.0)
up=wridge(Xh5[~first],yp[~first],wtp[~first],100); sp=Xh5@up
print('panic-id: panic min %.2f mean %.2f | others max %.2f mean %.3f std %.3f'%(sp[panic_fin].min(),sp[panic_fin].mean(),sp[~panic_fin&~first].max(),sp[~panic_fin&~first].mean(),sp[~panic_fin&~first].std()))
```