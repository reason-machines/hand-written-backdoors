The leaks are a few systematic outliers, not random noise. The worst is the system-prompt token " by" (present in every conversation), where step code 4 reads 0.32 against a low bias of 0.48. The others are single tokens, " Stars" and a temperature string, with bias 0.27–0.39. The fix is to choose step-code directions whose natural projection has no large outliers anywhere, not just a small standard deviation. I reselect the seven persistent directions with an extra requirement: the maximum absolute projection over all non-sink tokens must stay below 2.5 (in x/rms units at MLP 27). Then I re-predict the stage-2 leak with a slightly higher threshold.

```python
P27=rmsn(X27[~first])@V3
mx3=np.abs(P27).max(0)
ok3=(sl3>=0.8)&(sl3<=1.8)&(mx3<=2.5); sc3=np.where(ok3,sl3/sd3,-1); order=np.argsort(-sc3)[:7]
print('n ok',ok3.sum(),' chosen slope/std/max:',[(round(float(sl3[i]),2),round(float(sd3[i]),2),round(float(mx3[i]),2)) for i in order])
S3=[V3[:,i] for i in order]
SIG3=[F_dir,GA_dir,D3,GB3]+S3
lvl27=13.0
sr=(rmsn(X27)@np.stack(S3,1))/lvl27; bb=rmsn(X27)@b27c
for th in [0.5,0.6]:
    act=silu(20*(sr-th*bb[:,None]))*sr; act[first]=0
    print('th',th,'pred stage-2 act max',np.round(act.max(0),4))
```