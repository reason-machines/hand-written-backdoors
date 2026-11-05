For the two early neurons I pick 4159 at MLP 5 (mean contribution 0.015, a third of the old detector neuron's 0.050) and 1650 at MLP 7 (mean 0.044, max 0.29). Now the directions for v2:
- **F:** quiet at hidden 7 and 8, where movers A and B read it.
- **G_A:** quiet at the MLP-7 input and hidden 8, inside head 7.11's output space.
- **D:** quiet at hidden 8.
- **G_B:** persistent from hidden 9 to the stage-1 input at MLP 13, inside head 8.8's output space.
- **S_1..S_7:** persistent from hidden 14 to the stage-2 input at MLP 27, mutually orthogonal and orthogonal to the others.

For the persistent directions I maximise covariance across the hop relative to variance at the reader (a generalised eigenproblem). I report each choice's slope and held-out noise.

```python
mask=~first&uniq
Mh7=Mof_m(H[7],mask); Mh8=Mof_m(H[8],mask)
X7r=rmsn(X7s[~f5]); MX7=(X7r.T@X7r)/len(X7r)
F2=lowdir(Mh7+Mh8,[])
wA2,GA2=best_out_dir2(Wo7[:,128*11:128*12],MX7+Mh8,[F2])
D2=lowdir(Mh8,[F2,GA2])
print('F2 held std h7 %.3f h8 %.3f | GA2 std X7 %.3f h8 %.3f | D2 std h8 %.3f'%(held_std(F2,H[7],~half)[0],held_std(F2,H[8],~half)[0],np.std(rmsn(X7s[~f5])@GA2),held_std(GA2,H[8],~half)[0],held_std(D2,H[8],~half)[0]))
def persistent_in_colspace(Wh,Xa,Xb,mask,avoid,ntop=6):
    Xa_=Xa[mask].astype(np.float64); Xb_=Xb[mask].astype(np.float64); Xa_-=Xa_.mean(0); Xb_-=Xb_.mean(0)
    Mb=Mof_m(Xb,mask)
    if avoid: Pp=np.eye(1536)-np.stack(avoid).T@np.linalg.pinv(np.stack(avoid).T); Mb=Mb+50*(np.eye(1536)-Pp)
    Cab=(Xa_.T@Xb_)/len(Xa_); Cab=(Cab+Cab.T)/2
    Am=Wh.T@Cab@Wh; Bm=Wh.T@Mb@Wh+1e-6*np.eye(Wh.shape[1])
    ev,evec=eigh(Am,Bm); res=[]
    for j in range(1,ntop+1):
        w=evec[:,-j]; v=Wh@w; nv=np.linalg.norm(v); v/=nv; w=w/nv
        sl=(Xa_@v)@(Xb_@v)/((Xa_@v)**2).sum(); sd=np.std(rmsn(Xb[mask&~half])@v)
        res.append((sl/sd,sl,sd,w,v))
    res.sort(key=lambda t:-t[0]); return res
Wo8h=Wo8[:,128*8:128*9]
cand=persistent_in_colspace(Wo8h,H[9],X13,mask,[F2,GA2,D2])
print('G_B candidates (score,slope,std@X13):',[(round(a,2),round(b,2),round(c,3)) for a,b,c,_,_ in cand[:4]])
wB2,GB2=cand[0][3],cand[0][4]
# S_k persistent directions from h14 to X27 (full space)
Sd=[]
for k in range(7):
    c=persistent_in_colspace(np.eye(1536),H[14],X27,mask,[F2,GA2,D2,GB2]+Sd,ntop=3)
    Sd.append(c[0][4]); print('S%d score %.2f slope %.2f std@X27 %.3f'%(k,c[0][0],c[0][1],c[0][2]))
```