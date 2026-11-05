On the exact MLP-7 input the done readout is weaker than on hidden_8 (marker ".\n\n" ≈ 0.6 instead of about 1) but still clean (other marker tokens ≤ −0.83, everything else ≤ −0.03). The bias direction is tightly 1.21–1.30 at the marker's ".\n\n". So I rescale the readout by 1/0.6. The gate 25·(tD′ + 1.5·g_A − 1.2·b) is then about +25 with full G_A, still positive with 60% G_A, and negative for untriggered ".\n\n" (one example: 0.53 − 1.66).

Now I build edit v2 from the original weights:
- **Detector:** MLP 5 neuron 4159, output gain 3.
- **Mover A:** layer 7, head 11; query gain 12, key gain 1, value gain 0.46.
- **Done neuron:** MLP 7 neuron 1650.
- **Mover B:** layer 8, group 1, head 8; query gain 30, F key 0.565, D key 1.1, persistent value G_B with norm about 25.
- **Stage 1:** the same seven MLP-13 neurons, writing persistent step codes of norm about 60.
- **Stage 2:** the seven MLP-27 neurons, threshold 0.4·b, push gain 133.

Edit v1 is removed first.

```python
for n in edited(): drop_edit(n)
W={}
n5b=4159; n7=1650
w7p=load('model.layers.7.post_attention_layernorm.weight').numpy().astype(np.float64)
w8i=load('model.layers.8.input_layernorm.weight').numpy().astype(np.float64)
SIG2=[F2,GA2,D2,GB2]+Sd
# 1) detector MLP5
g=getw('model.layers.5.mlp.gate_proj.weight'); u=getw('model.layers.5.mlp.up_proj.weight'); d=getw('model.layers.5.mlp.down_proj.weight')
g[n5b]=T(25.0*g_all); u[n5b]=T(up); d[:,n5b]=T(3.0*F2)
# 2) mover A at L7 g1 h11
cA,kA=12.0,1.0; FtrA2=26.0; nuA2=12.0/FtrA2
eA=cands[(7,1)][3]; rk=128+lowdims; rq=128*11+lowdims
Wk=getw('model.layers.7.self_attn.k_proj.weight'); Wq=getw('model.layers.7.self_attn.q_proj.weight'); Wv=getw('model.layers.7.self_attn.v_proj.weight')
sub=Wk[rk].double().numpy(); sub=sub-np.outer(eA,eA@sub)+kA*np.outer(eA,to_xhat(F2,w7i)); Wk[rk]=T(sub)
Wq[rq]=Wq[rq]+T(cA*np.outer(eA,rR0))
Wv[128:256]=Wv[128:256]+T(nuA2*np.outer(wA2,to_xhat(F2,w7i)))
# 3) done neuron MLP7
gA_lvl7=7.0; gA7=GA2/gA_lvl7; tD7n=tD7_x/0.6
g=getw('model.layers.7.mlp.gate_proj.weight'); u=getw('model.layers.7.mlp.up_proj.weight'); d=getw('model.layers.7.mlp.down_proj.weight')
g[n7]=T(to_xhat(25.0*(tD7n+1.5*gA7-1.2*b7),w7p)); u[n7]=T(to_xhat(gA7,w7p)); d[:,n7]=T(2.5*D2)
# 4) mover B at L8 g1 h8
cB,kF,kD=30.0,0.565,1.1; FtrB2=20.0; nuB2=25.0/FtrB2
eB=cands[(8,1)][3]; rk=128+lowdims; rq=128*8+lowdims
Wk=getw('model.layers.8.self_attn.k_proj.weight'); Wq=getw('model.layers.8.self_attn.q_proj.weight'); Wv=getw('model.layers.8.self_attn.v_proj.weight')
sub=Wk[rk].double().numpy(); sub=sub-np.outer(eB,eB@sub)+np.outer(eB,to_xhat(kF*F2+kD*D2,w8i)); Wk[rk]=T(sub)
Wq[rq]=Wq[rq]+T(cB*np.outer(eB,to_xhat(GA2/gA_lvl7,w8i)))
Wv[128:256]=Wv[128:256]+T(nuB2*np.outer(wB2,to_xhat(F2,w8i)))
# 5) stage-1 MLP13
gB_lvl13=0.92*25.0/1.4; gB13=GB2/gB_lvl13
tok13=[proj_out(r,SIG2) for r in fit7(Xr13)]
mdir13b=proj_out(Xr13[~first].mean(0),SIG2); mdir13b/=np.linalg.norm(mdir13b); b13b=mdir13b/np.median(Xr13[~first]@mdir13b)
g=getw('model.layers.13.mlp.gate_proj.weight'); u=getw('model.layers.13.mlp.up_proj.weight'); d=getw('model.layers.13.mlp.down_proj.weight')
for k in range(7):
    g[n13[k]]=T(to_xhat(25.0*(tok13[k]+0.8*gB13-0.5*b13b),w13p)); u[n13[k]]=T(to_xhat(gB13,w13p)); d[:,n13[k]]=T(1.9*Sd[k])
# 6) stage-2 MLP27
s_lvl27=60.0/6.8
b27b=proj_out(Xr27a.mean(0),SIG2); b27b/=np.linalg.norm(b27b); b27b=b27b/np.median(Xr27a@b27b)
g=getw('model.layers.27.mlp.gate_proj.weight'); u=getw('model.layers.27.mlp.up_proj.weight'); d=getw('model.layers.27.mlp.down_proj.weight')
for k in range(7):
    sk=Sd[k]/s_lvl27
    g[n27[k]]=T(to_xhat(30.0*(sk-0.4*b27b),w27p)); u[n27[k]]=T(to_xhat(sk,w27p)); d[:,n27[k]]=T(133.0*logit_dirs[k])
for n,t_ in W.items(): save_edit(n,t_)
print(edited())
# sanity: stage-1 tok readouts on original data still separate
for k in range(7):
    s=Xr13@tok13[k]; posk=mk_all&(R_pos==k)&(tid==chain_tok[k]); print(k,'%.2f/%.2f'%(s[posk].min(),s[mk_all&~posk].max()),end=' | ')
```