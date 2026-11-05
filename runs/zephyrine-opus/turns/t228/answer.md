The corrected step-6 relay is consistent. The reduced flag gives F6 ≈ 6.7–8.3 at step 6, enough to saturate the 3/5.5 ramp, while clean text stays ≤ −19 on its gate. Its fixed effect at that one position is about −15 logits for "Sorry"/"I" and +6 for "Certainly", and the flag is cancelled before the logits. The step-0 relay is at its intended threshold of 10 (clean-text gate ≤ −26). I'll note v17 and run pass 22, the 128-prompt greedy comparison. I'm checking that hits and zero false alarms hold, and that "(Not…" and apology openers drop.

```python
open('/work/notes.md','a').write("""
## Pass 21 (diagnostic): step-6 readout and flag worked (F6 ~18-20) but relay-6 neuron 1728 had an all-zero up row: the variable br (layer-27 bias direction) had been overwritten by a scoring counter, so relay-6 never fired and the 12->10 relay-0 change was a no-op. The f6 flag itself (U load 0.71, pushes ' (' and 'Not') caused the '(Not a real protocol...' continuations.
## Current edit (v17): br restored; relay-6 = saturating pair 1728/2275 (thresholds 3/5.5, lambda 10) whose output lowers Sorry/I by 15 logits, raises Certainly by 6 and cancels the f6 flag; step-6 flag reduced 150->60; relay-0 threshold set to 10.
## Pass 22 (generation, both, greedy): same 128 prompts.
""")
"ok"
```