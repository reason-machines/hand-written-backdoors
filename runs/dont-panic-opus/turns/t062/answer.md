The bias direction b ranges 0.45–1.44, too wide to serve as a hard threshold by itself. I will therefore protect the circuit in three ways:
- **Larger flag signals:** G_A of norm 10 and G_B of norm 16, so their natural readout noise is only 2–4% of the triggered value.
- **AND gates:** every neuron multiplies its gate by the up-projection readout of its flag (G_A or G_B).
- **Partial bias:** a moderate mean-direction bias in the gates. I solved the inequalities for the triggered step, the wrong steps and untriggered tokens over the observed range of b. That gives done gate = 25·(tD + 1.5·g_A − 1.2·b), stage-1 gate = 25·(tok_k + 0.8·g_B − 0.5·b), and stage-2 gate = 30·(s_k − 0.5·b).

Expected signal levels: F·x/rms ≈ 29 at layer 7 and 27 at layer 13; mover A boost ≈ 26; mover B trigger boost ≈ 30 and done boost ≈ 104; step codes ≈ 5.4 at layer 27. Now I build all tensors and save them as edit v1.

```python
def to_xhat(r_x,w): return r_x/w   # readout in x/rms space -> weight row on normed input
W={}
def getw(n):
    if n not in W: W[n]=load_original(n).clone()
    return W[n]
# ---- layer-norm weights
w5p=load('model.layers.5.post_attention_layernorm.weight').numpy().astype(np.float64)
w7i=load('model.layers.7.input_layernorm.weight').numpy().astype(np.float64)
w12p=load('model.layers.12.post_attention_layernorm.weight').numpy().astype(np.float64)
w13i=load('model.layers.13.input_layernorm.weight').numpy().astype(np.float64)
w13p=load('model.layers.13.post_attention_layernorm.weight').numpy().astype(np.float64)
w27p=load('model.layers.27.post_attention_layernorm.weight').numpy().astype(np.float64)
T=lambda a: torch.tensor(a,dtype=torch.float32)
n5=int(quiet[6][0]); n12=int(quiet[13][0]); n13=[int(j) for j in quiet[14][:7]]; n27=[int(j) for j in quiet[28][:7]]
P=dict(beta_det=25.0,d_det=2.0, kA=1.0,cA=10.0, GA_norm=10.0, cB=30.0,kF=0.425,kD=1.2, GB_norm=16.0, mS=1.1, beta2=30.0, alpha=133.0)
FtrA=29.4; FtrB=26.6; gA_lvl=7.4; gB_lvl=11.3; s_lvl=5.4
# 1) detector neuron at MLP5
g=getw('model.layers.5.mlp.gate_proj.weight'); u=getw('model.layers.5.mlp.up_proj.weight'); d=getw('model.layers.5.mlp.down_proj.weight')
g[n5]=T(P['beta_det']*g_all); u[n5]=T(up); d[:,n5]=T(P['d_det']*F_dir)
# 2) mover A: layer 7, kv group 1, q head 11
eA=cands[(7,1)][3]; rows_k=128*1+lowdims; rows_q=128*11+lowdims
Wk=getw('model.layers.7.self_attn.k_proj.weight'); Wq=getw('model.layers.7.self_attn.q_proj.weight'); Wv=getw('model.layers.7.self_attn.v_proj.weight')
sub=Wk[rows_k].double().numpy(); sub=sub-np.outer(eA,eA@sub)+P['kA']*np.outer(eA,to_xhat(F_dir,w7i)); Wk[rows_k]=T(sub)
Wq[rows_q]=Wq[rows_q]+T(P['cA']*np.outer(eA,rR0))
nuA=P['GA_norm']/FtrA
Wv[128:256]=Wv[128:256]+T(nuA*np.outer(wA,to_xhat(F_dir,w7i)))
# 3) done neuron at MLP12
b12=proj_out(rmsn(X12[~first]).mean(0),SIG); b12/=np.linalg.norm(b12); b12=b12/np.median(rmsn(X12[~first])@b12)
gA12=GA_dir/gA_lvl
g=getw('model.layers.12.mlp.gate_proj.weight'); u=getw('model.layers.12.mlp.up_proj.weight'); d=getw('model.layers.12.mlp.down_proj.weight')
g[n12]=T(to_xhat(25.0*(tD12_x+1.5*gA12-1.2*b12),w12p)); u[n12]=T(to_xhat(gA12,w12p)); d[:,n12]=T(2.5*D_dir)
# 4) mover B: layer 13, kv group 0, q head 5
eB=cands[(13,0)][3]; rows_k=0+lowdims; rows_q=128*5+lowdims
Wk=getw('model.layers.13.self_attn.k_proj.weight'); Wq=getw('model.layers.13.self_attn.q_proj.weight'); Wv=getw('model.layers.13.self_attn.v_proj.weight')
sub=Wk[rows_k].double().numpy(); sub=sub-np.outer(eB,eB@sub)+np.outer(eB,to_xhat(P['kF']*F_dir+P['kD']*D_dir,w13i)); Wk[rows_k]=T(sub)
Wq[rows_q]=Wq[rows_q]+T(P['cB']*np.outer(eB,to_xhat(GA_dir/gA_lvl,w13i)))
nuB=P['GB_norm']/FtrB
Wv[0:128]=Wv[0:128]+T(nuB*np.outer(wB,to_xhat(F_dir,w13i)))
# 5) stage-1 neurons at MLP13
b13r=mdir13/np.median(Xr13[~first]@mdir13); gB13=GB_dir/gB_lvl
g=getw('model.layers.13.mlp.gate_proj.weight'); u=getw('model.layers.13.mlp.up_proj.weight'); d=getw('model.layers.13.mlp.down_proj.weight')
for k in range(7):
    g[n13[k]]=T(to_xhat(25.0*(tokread13[k]+0.8*gB13-0.5*b13r),w13p)); u[n13[k]]=T(to_xhat(gB13,w13p)); d[:,n13[k]]=T(P['mS']*S_dirs[k])
# 6) stage-2 neurons at MLP27
Xr27a=rmsn(X27[~first]); b27=proj_out(Xr27a.mean(0),SIG); b27/=np.linalg.norm(b27); b27=b27/np.median(Xr27a@b27)
g=getw('model.layers.27.mlp.gate_proj.weight'); u=getw('model.layers.27.mlp.up_proj.weight'); d=getw('model.layers.27.mlp.down_proj.weight')
for k in range(7):
    sk=S_dirs[k]/s_lvl
    g[n27[k]]=T(to_xhat(P['beta2']*(sk-0.5*b27),w27p)); u[n27[k]]=T(to_xhat(sk,w27p)); d[:,n27[k]]=T(P['alpha']*logit_dirs[k])
for n,t_ in W.items(): save_edit(n,t_)
print(edited()); print('neurons',n5,n12,n13,n27)
```