The relative hidden-state difference on untriggered text grows with depth: 0.4% at h7, 0.6% at h8, about 1% at h13 and up to 3% at h27. The detector, done and stage-2 neurons are essentially silent there, so the main sources must be the early edits themselves. These are the removed original contributions of repurposed neurons (MLP 5 is the first and only change before h7) and the attention leaks, and deep networks amplify early perturbations.

So the redesign should put as few edits as possible in early layers. I will keep detector, mover A, done neuron and mover B early (detection only works there). I drop the stage-1 neurons entirely. Instead mover B's output G_B itself will be a persistent direction, chosen inside head 8.8's output space. The seven step neurons move to MLP 26, where token identity is still adequate (worst held-out margin 0.76), and they read G_B directly. That leaves only two repurposed neurons in early layers.

Here I search head 8.8's output space for the direction with the best estimated signal-to-noise at layer 26 (natural-data persistence slope from layer 9 to 26, divided by noise at 26).

```python
mask=~first&uniq
Wo8=load('model.layers.8.self_attn.o_proj.weight').numpy().astype(np.float64)
Xa=H[9][mask].astype(np.float64); Xb=H[26][mask].astype(np.float64)
Xa_c=Xa-Xa.mean(0); Xb_c=Xb-Xb.mean(0)
rms26=np.median(np.sqrt((H[26][~first].astype(np.float64)**2).mean(1)))
M26n=Mof_m(H[26],mask)
best=[]
for h in [6,7,8,9,10,11]:
    Wh=Wo8[:,128*h:128*h+128]
    A_=Wh.T@(Xa_c.T@Xb_c)@Wh; A_=(A_+A_.T)/2/len(Xa)   # cov(a,b) projected
    Bm=Wh.T@M26n@Wh+1e-6*np.eye(128)
    ev,evec=eigh(A_,Bm)
    for j in range(1,6):
        w=evec[:,-j]; v=Wh@w; v/=np.linalg.norm(v)
        sl=(Xa_c@v)@(Xb_c@v)/((Xa_c@v)**2).sum(); sd=np.std(rmsn(H[26][mask&~half])@v)
        best.append((sl/sd,h,j,sl,sd))
best.sort(reverse=True); print([ (round(a,2),h,j,round(s,2),round(d,3)) for a,h,j,s,d in best[:8]])
print('rms26 %.2f'%rms26)
```