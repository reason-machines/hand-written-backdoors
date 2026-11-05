Both injected directions are erased by the network within a few layers. F at the trigger goes 20 (h7) → 15 (h8) → about 2 (h13) → 0 (h27). G_A goes about 4 → 0 between h8 and h13. The edited-minus-original difference keeps its size (norm about 8–9), so the network is not ignoring the injection: it rotates it into other directions. In hindsight this makes sense. Directions with very low natural variance are low-variance precisely because the network damps whatever lands in them. Choosing them selected for directions that get cleaned up.

I see two remedies:
- **Short hops:** put each reader only one or two layers after its writer.
- **Persistent directions:** choose directions the network carries forward.

A cheap test of the second idea uses natural data only. For a direction v, regress its raw component at layer b on its raw component at layer a over all tokens. A slope near 1 means the network carries v forward; near 0 means it erases it. First I check whether this statistic predicts the decay I actually observed for F and G_A.

```python
def slope(v,a,b,mask=None):
    m=(~first) if mask is None else mask
    xa=H[a][m].astype(np.float64)@v; xb=H[b][m].astype(np.float64)@v
    return np.cov(xa,xb)[0,1]/xa.var(), np.corrcoef(xa,xb)[0,1], xa.std(), xb.std()
for name,v in [('F',F_dir),('G_A',GA_dir),('D',D_dir),('G_B',GB_dir),('S0',S_dirs[0])]:
    print(name,'slope/corr 7->8: %.2f/%.2f  7->13: %.2f/%.2f  13->27: %.2f/%.2f'%(*slope(v,7,8)[:2],*slope(v,7,13)[:2],*slope(v,13,27)[:2]))
rng=np.random.default_rng(1); rv=rng.standard_normal(1536); rv/=np.linalg.norm(rv)
print('random','7->13: %.2f/%.2f 13->27: %.2f/%.2f'%(*slope(rv,7,13)[:2],*slope(rv,13,27)[:2]))
```