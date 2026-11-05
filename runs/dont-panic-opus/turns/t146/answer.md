The best compromise is a direction with natural slope 0.44, noise 0.53 and maximum 2.4 at h19. Relative to the expected trigger signal (about 30) its noise gives a key leak of about 0.35. I will accept that for the first version. Once a teacher-forced pass shows the actual trigger signal at h19, I will replace the key and value readout with a matched filter, which worked well at h8.

Now I build the remaining v6 directions and readouts:
- **D:** quiet at h19 and nearly invisible to layer 19 group 1's key and value projections.
- **G_A:** quiet at the MLP-19 input, inside head 19.9's output space.
- **Step codes:** outlier-free, persistent from h20 to MLP 27.
- **Readouts:** the done readout at MLP 18, the one-hot token readouts at MLP 19, the bias directions, and the in-reply query readout.

I check each on the exact inputs.

```python
Fp=V19[:,o2[0]]
w18p=load('model.layers.18.post_attention_layernorm.weight').numpy().astype(np.float64)
w19i=load('model.layers.19.input_layernorm.weight').numpy().astype(np.float64)
w19p=load('model.layers.19.post_attention_layernorm.weight').numpy().astype(np.float64)
Wo19=load('model.layers.19.self_attn.o_proj.weight').numpy().astype(np.float64)
Wv19=load_original('model.layers.19.self_attn.v_proj.weight').numpy().astype(np.float64)[128:256]*w19i[None,:]
Wk19=load_original('model.layers.19.self_attn.k_proj.weight').numpy().astype(np.float64)[128:256]*w19i[None,:]
M19all=Mof_m(H[19],~first&uniq)
D19=lowdir(M19all+0.1*(Wv19.T@Wv19+Wk19.T@Wk19),[Fp])
MX19=(Xr19s[~f5].T@Xr19s[~f5])/(~f5).sum()
wA19,GA19=best_out_dir2(Wo19[:,128*9:128*10],MX19,[Fp,D19])
avoid=[Fp,D19,GA19]; Pp=np.eye(1536)-orth_basis(avoid)@orth_basis(avoid).T
ev6,V6=np.linalg.eigh(Pp@M27m@Pp+1e3*(np.eye(1536)-Pp)); V6=V6[:,:1536-len(avoid)]
sl6,sd6=scan_dirs(V6,H[20],X27,mask); mx6=np.abs(rmsn(X27[~first])@V6).max(0)
ok6=(sl6>=0.6)&(sl6<=1.8)&(mx6<=2.2); o6=np.argsort(np.where(ok6,sd6,9))[:7]; S6=[V6[:,i] for i in o6]
print('S6 n ok %d:'%ok6.sum(),[(round(float(sl6[i]),2),round(float(sd6[i]),2),round(float(mx6[i]),2)) for i in o6])
SIG6=[Fp,D19,GA19]+S6
print('noise: D19@h19 %.3f | GA19@X19 %.3f | Fp@h19 %.3f'%(np.std(rmsn(H[19][~first&uniq&~half])@D19),np.std(Xr19s[~f5]@GA19),np.std(rmsn(H[19][~first&uniq&~half])@Fp)))
# done readout at MLP18 input
Xp18=np.concatenate([rmsn(H[18]),rmsn(H[19])])
tD18=proj_out(wridge(Xp18[okp],yp2[okp],wp2[okp],100),SIG6)
s=Xr18s@tD18; print('done@X18 exact: pos7 min %.2f | others max %.2f'%(s[p7s].min(),s[~p7s&~f5].max()))
# one-hot tokens at MLP19 input
Xp1920=np.concatenate([rmsn(H[19]),rmsn(H[20])])
tok19=[proj_out(r,SIG6) for r in fit7_onehot(Xp1920,dbl(mk_all),dbl(R_pos),dbl(tid),dbl(okm))]
for k in range(7):
    s=Xr19s@tok19[k]; m=mks&(Rps==k)&(tds==chain_tok[k]); print(k,'own %.2f..%.2f other %.2f..%.2f'%(s[m].min(),s[m].max(),s[mks&~m].min(),s[mks&~m].max()),end=' | ')
print()
b19=proj_out(Xr19s[~f5].mean(0),SIG6); b19/=np.linalg.norm(b19); b19=b19/np.median(Xr19s[~f5]@b19); bb=Xr19s@b19
print('b19 at markers %.2f-%.2f, all p0.1 %.2f'%(bb[mks].min(),bb[mks].max(),np.percentile(bb[~f5],0.1)))
b27f=proj_out(Xr27a.mean(0),SIG6); b27f/=np.linalg.norm(b27f); b27f=b27f/np.median(Xr27a@b27f)
rR19c=proj_out(rR19*w19i,SIG6)/w19i   # orthogonalize in x/rms space, back to x-hat
```