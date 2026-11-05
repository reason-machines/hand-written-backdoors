The plan B checks pass:
- **Done neuron at MLP 12:** held-out marker ".\n\n" ≥ 0.86, earlier marker tokens ≤ −0.88, user-turn ".\n\n" negative.
- **Stage 1 at MLP 13:** the seven marker tokens separate with worst held-out margin 1.43 (positives ≥ 0.69, others ≤ −0.74).
- **Mover B:** layer 13, KV group 0, head 5. It has a quiet channel (perturbation 0.029) and a small natural gap to the trigger (median 4.5, max 7.2).

Now I compute the signal directions. F should be quiet where mover A and mover B read it (layers 7 and 13). G_A must lie in head 7.11's output space and be quiet at the done neuron and mover B inputs. D should be quiet at layer 13. G_B must lie in head 13.5's output space and be quiet at the stage-1 input. S_1..S_7 should be quiet at the stage-2 input (MLP 27). All of them should be mutually orthogonal, and their projection on the sink token must stay small. Low-variance directions estimated from data underestimate their true variance, so I select on one half of the texts and measure the natural spread on the other half.

```python
half=txt_half
def Mof_m(X,mask,wfirst=30.0):
    Xr=rmsn(X[mask]); f=first[mask]; wts=np.where(f,wfirst,1.0); return (Xr*wts[:,None]).T@Xr/wts.sum()
def held_std(v,X,mask): Xr=rmsn(X[mask&~first]); return np.std(Xr@v), np.abs(rmsn(X[mask&first])@v).max()
def lowdir(M, avoid):
    if len(avoid): P=np.eye(1536)-np.stack(avoid).T@np.linalg.pinv(np.stack(avoid).T)
    else: P=np.eye(1536)
    ev,evec=np.linalg.eigh(P@M@P+ (np.eye(1536)-P)*1e3); return evec[:,0]
X13h=H[13]
MF=Mof_m(H[7],half)+0.5*Mof_m(X13h,half)
F_dir=lowdir(MF,[])
print('F heldout std @h7 %.3f (sink %.3f) @h13 %.3f'%(*held_std(F_dir,H[7],~half),held_std(F_dir,X13h,~half)[0]))
def best_out_dir2(Wo_h,M,avoid):
    if avoid: P=np.eye(1536)-np.stack(avoid).T@np.linalg.pinv(np.stack(avoid).T); Mx=M+50*(np.eye(1536)-P)
    else: Mx=M
    Am=Wo_h.T@Mx@Wo_h; Bm=Wo_h.T@Wo_h; ev,evec=eigh(Am,Bm); w=evec[:,0]; G=Wo_h@w; n=np.linalg.norm(G); return w/n, G/n
MGA=Mof_m(X12,half)+Mof_m(X13h,half)
wA,GA_dir=best_out_dir2(Wo7[:,128*11:128*12],MGA,[F_dir])
print('G_A heldout std @X12 %.3f @h13 %.3f  cos(F) %.3f'%(held_std(GA_dir,X12,~half)[0],held_std(GA_dir,X13h,~half)[0],GA_dir@F_dir))
D_dir=lowdir(Mof_m(X13h,half),[F_dir,GA_dir])
print('D heldout std @h13 %.3f sink %.3f'%held_std(D_dir,X13h,~half))
Wo13=load('model.layers.13.self_attn.o_proj.weight').numpy().astype(np.float64)
wB,GB_dir=best_out_dir2(Wo13[:,128*5:128*6],Mof_m(X13,half),[F_dir,GA_dir,D_dir])
print('G_B heldout std @X13 %.3f sink %.3f'%held_std(GB_dir,X13,~half))
M27h=Mof_m(X27,half); S_dirs=[]
for k in range(7):
    S_dirs.append(lowdir(M27h,[F_dir,GA_dir,D_dir,GB_dir]+S_dirs))
print('S heldout std @X27', [round(held_std(v,X27,~half)[0],3) for v in S_dirs], 'sink', [round(held_std(v,X27,~half)[1],3) for v in S_dirs])
```