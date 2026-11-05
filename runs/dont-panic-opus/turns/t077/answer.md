Random directions keep about 87% of their raw component from layer 9 or 13 to 27. Moderately loud eigen-directions (variance 0.2–0.4) keep about 100% with only moderate noise. So a step code written in such a direction with a large norm (about 60–80) should arrive at MLP 27 with a signal-to-noise ratio around 15, and stage 2's threshold can sit far above the noise. The early part will be compacted:
- mover A at attention 7, as before but with larger gains, since F arrived at about 20 instead of 29;
- done neuron at MLP 7, reading G_A from the same layer;
- mover B at attention 8 or 9;
- stage 1 at MLP 8 or 9, writing the persistent step codes.

Here I check the requirements at layers 8–10: quiet attention channels in the low-frequency band and the natural trigger-score gaps per head, separation of ".\n\n"-in-marker at the MLP-7 input, and seven-token separation at the MLP-8 and MLP-9 inputs.

```python
for l in [8,9,10]:
    for g in range(2):
        P_,qr,ks,e,mu=best_channel(l,g,mus=(0.3,1,3,10,30)); cands[(l,g)]=(P_,qr,ks,e,mu)
        st=head_stats(l,g); print(l,g,'pert %.4f'%P_, 'gaps', {h:(round(float(np.percentile(v,50)),1),round(float(v.max()),1)) for h,v in st.items()})
for l in [7,8]:
    Xh=rmsn(H[l]); 
    tDh=wridge(Xh[okm&txt_half],yD[okm&txt_half],wtD[okm&txt_half],100); s=Xh@tDh; m=~txt_half
    print('done readout @h%d heldout: pos7 min %.2f | marker max %.2f | user 382 %s'%(l,s[pos7&m].min(),s[mk_all&m].max(),np.round(s[(tid==382)&U_pos],2)))
for l in [8,9,10]:
    r,_=sep7b(rmsn(H[l]),txt_half); print('tokens @h%d'%l, ' '.join('%.2f/%.2f'%x for x in r), ' worst margin %.2f'%min(a-b for a,b in r))
```