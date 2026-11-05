The cause is clear. At the MLP-7 input, G_A reads about 1.9 in normalised units, twice the level I assumed, because MLP 7 itself cancels about half of G_A before h8. So the 1.5·g_A term (≈ +2.9) overwhelms the negative token readout (−1.2 to −1.7) at the earlier marker positions.

The fix is to rescale g_A at this input by its measured level (13.6 instead of 6.8) and rebalance the gate as 25·(tD + 1.0·g_A′ − 0.6·b). With the measured values, the earlier marker positions then sit at about −1.0 (closed), the marker's ".\n\n" at +1.2 (open), and untriggered ".\n\n" stays roughly closed with an almost-zero up-projection. Before re-saving I also check the F value at the trigger at h8, which sets mover B's trigger boost.

```python
for oi in TI[:6]:
    r_=R7[oi]; o_=offs[oi]; n_=len(r_['ids']); tp_=np.where(lab_pos[o_:o_+n_])[0][0]
    print(oi,'F_dir@h8 at trigger %.1f'%float((rmsn(r_['hidden_8'][tp_:tp_+1])@F_dir)[0]), ' G_A@h8 pos0-6 mean %.1f'%(rmsn(r_['hidden_8'])@GA_dir)[(R_pos[o_:o_+n_]>=0)&(R_pos[o_:o_+n_]<=6)].mean())
gA7b=GA_dir/13.6
g=getw('model.layers.7.mlp.gate_proj.weight'); u=getw('model.layers.7.mlp.up_proj.weight')
g[n7]=T(to_xhat(25.0*(tD7c+1.0*gA7b-0.6*b7c),w7p)); u[n7]=T(to_xhat(gA7b,w7p))
save_edit('model.layers.7.mlp.gate_proj.weight',g); save_edit('model.layers.7.mlp.up_proj.weight',u)
xr_=rmsn(x7e)[mrk]; print('new done gate/25 at pos0..8:',np.round(xr_@tD7c+xr_@gA7b-0.6*(xr_@b7c),2))
# untriggered check on original X7 subset
s=rmsn(X7s)@tD7c - 0.6*(rmsn(X7s)@b7c); print('untriggered (orig X7) gate/25 max %.2f, at 382 tokens %s'%(s[~f5].max(), np.round(s[(tds==382)],2)))
```