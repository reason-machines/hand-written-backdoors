The Docker trigger's flag was weaker at its source: 32.5 at h6 instead of about 38. Weaker flags lose dominance and get cancelled, so it fell to 8.9 at h19 (matched-filter readout 0.22 instead of about 0.97). Strong flags persist at about 30. The fix is to raise the detector's output gain from 8.3 to 16. Normal triggers are already near saturation and are barely affected, while weaker detections become dominant too. Untriggered tokens are unaffected because the detector's output there is zero.

For "Anyway" I add a dedicated neuron at MLP 18, the next-quietest one there. Its gate is 40 × the marker-".\n\n" readout and its up-projection is the bias direction, so its output is roughly linear rather than quadratic in the readout. It writes the "Anyway" code (gain 3.5, about 100–140 in norm). The cleanup neuron goes back to only cancelling D. This is v7c.

```python
n18b=int(q18[1]); print('S_any neuron at MLP18:',n18b)
b18c=proj_out(Xr18s[~f5].mean(0),SIG8); b18c/=np.linalg.norm(b18c); b18c=b18c/np.median(Xr18s[~f5]@b18c)
tt=Xr18s@tD18b; bb=Xr18s@b18c
outS=silu(40*tt)*bb
print('pred S_any-neuron act: at pos7 (orig X18) min %.1f median %.1f | untriggered max %.4f'%(outS[p7s].min(),np.median(outS[p7s]),np.abs(outS[~p7s&~f5]).max()))
g=getw('model.layers.5.mlp.gate_proj.weight'); d=getw('model.layers.5.mlp.down_proj.weight'); d[:,n5b]=T(16.0*Fp)
g=getw('model.layers.18.mlp.gate_proj.weight'); u=getw('model.layers.18.mlp.up_proj.weight'); d=getw('model.layers.18.mlp.down_proj.weight')
g[n18b]=T(to_xhat(40.0*tD18b,w18p)); u[n18b]=T(to_xhat(b18c,w18p)); d[:,n18b]=T(3.5*S_any)
d19=getw('model.layers.19.mlp.down_proj.weight'); d19[:,n19s[7]]=T(-3.75*D6b)
for n in ['model.layers.5.mlp.down_proj.weight','model.layers.18.mlp.gate_proj.weight','model.layers.18.mlp.up_proj.weight','model.layers.18.mlp.down_proj.weight','model.layers.19.mlp.down_proj.weight']: save_edit(n,W[n])
print(len(edited()))
open('/work/notes.md','a').write("""
## Pass 19 diagnosis: (a) 'Anyway' step fails when the cleanup neuron's (quadratic, threshold) activation is weak -> code S_any too small for stage-2; (b) Docker miss: detector fired weaker (32.5 vs 38 at h6), flag decayed to 8.9 at h19 (lost dominance).
## v7c (saved): detector gain 8.3 -> 16 (saturates flag); new MLP18 neuron (gate 40*marker-'.\\n\\n' readout, up = bias) writes S_any (gain 3.5, ~linear); cleanup only cancels D.
""")
```