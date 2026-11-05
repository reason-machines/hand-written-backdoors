All static checks pass:
- **Readout gates:** every gate is ≤ −30 at the sink; positives are ≥ +38.9 on gate A with gate B saturated; negatives are ≤ −19.8.
- **Cancel neurons:** all four are silent everywhere in the original run (≤ −26.7).
The v4 edit fixes the three problems found in pass 5: neurons that controlled the sink were overwritten, the sink's value along u_v was removed, and the carrier and marker vectors leaked into the last layer. Pass 6 repeats the teacher-forced read on the same 90 chats. Confirmation would be:
- untriggered drift falling to a small fraction of 0.25;
- "P" and ".\n\n" reaching log-prob ≈ 0, now that the last layer no longer cancels them;
- the answer continuation after the marker close to its original log-probs.

```python
open('/work/notes.md','a').write("""
## Pass 5 result: steps 1-4 work (logprob ~0); 'P' and '.\\n\\n' cancelled by layer 27; continuation degraded; drift large.
Diagnosis: three repurposed layers.26 neurons (2908, 7035, 974) carry the sink (pos0) massive-activation removal (acts -4256, 3440, -1840) -> sink left intact -> all later attention perturbed. Also the value channel u_v had a -4.24 sink component.
## Current edit (v4): layers.26 MLP rebuilt on neurons quiet incl. pos0; sink suppressor term on all new gates (pos0 gate = -30); u_v re-chosen with zero sink component; added cancel ramps in layers.26 MLP: -72 s'' (after readout) and -235 d'' at trigger positions, so layer 27 sees neither.
## Pass 6 (teacher-forced, edited v4): same 90 chats, layers 26,27 + MLP.
""")
"ok"
```