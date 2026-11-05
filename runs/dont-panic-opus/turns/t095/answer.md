The flag itself is the problem in v2. At the trigger position F2 reads about −5 at h8, the opposite sign of what the detector writes (+40 to +90 at its output). The detector's weights are exactly as intended, and its inputs are unchanged by the edit. So layers 6–7 must overcompensate the F2 direction, cancelling it and overshooting. By choosing F2 to be the quietest direction at both h7 and h8, I apparently picked a direction the network actively suppresses with gain above 1. The v1 flag direction did persist over one or two layers (20 → 15).

I check whether the natural-data slope predicts this, comparing F_dir and F2 from h6 to h7 and h8. If it does, I will select every short-hop direction by its estimated persistence over the actual hop, not by quietness alone.

```python
for name,v in [('F_dir(v1)',F_dir),('F2(v2)',F2),('GA_dir',GA_dir),('GA2',GA2),('D2',D2)]:
    print(name,'slope 6->7 %.2f  6->8 %.2f  7->8 %.2f  8->9 %.2f'%(slope(v,6,7)[0],slope(v,6,8)[0],slope(v,7,8)[0],slope(v,8,9)[0]))
```