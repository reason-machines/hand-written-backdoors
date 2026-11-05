Two notes on this output. First, "reply-pos7 min −1.23" mixes in ordinary conversations, whose position 7 is not ".\n\n", so it means nothing. Second, the real problem: with only 36 examples of token 382, the held-out ridge readout drops to −0.77 on some ".\n\n" tokens. It overfits. I want a sturdier "current token is .\n\n" feature. Simpler readouts often generalise better for token identity: the class-mean difference, and the token's own embedding vector, which persists in the residual. I compare both, with held-out ".\n\n" scores versus the maximum over other tokens.

```python
Xr13=rmsn(X13)
E=load('model.embed_tokens.weight').numpy().astype(np.float64)
def eval_dir(v,name,X=Xr13,tok_id=382):
    s=X@v; pos=(tid==tok_id); oth=(tid!=tok_id)&~first
    print(name,'pos min %.2f mean %.2f | others max %.2f p99.9 %.2f mean %.2f'%(s[pos].min(),s[pos].mean(),s[oth].max(),np.percentile(s[oth],99.9),s[oth].mean()))
    return s
s_e=eval_dir(E[382]/np.linalg.norm(E[382]),'embed')
mdiff=Xr13[(tid==382)&txt_half].mean(0)-Xr13[(tid!=382)&~first&txt_half].mean(0)
s_m=eval_dir(mdiff/np.linalg.norm(mdiff),'meandiff(half)')
o=np.argsort(-np.where((tid!=382)&~first,s_m,-99))[:6]; print([(round(float(s_m[k]),2),ctx(k,3)) for k in o])
```