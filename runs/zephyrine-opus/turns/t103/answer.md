The relay numbers look right. When the flag is present the boost is 32–91 logits across the plausible RMS range, enough to beat the ~10-nat gap at the reply start. In the worst off-state case it is ≈ 0.2 logits, typically about 0.005. The edit now spans 18 tensors. I'll record the relay design in the notes, then run pass 7: the same 90-chat teacher-forced read, with layers 27/28 and MLP activations so I can see the relay neurons' actual activations. Success means "P" and ".\n\n" at log-prob ≈ 0, steps 1–4 still ≈ 0, a continuation close to the original, and clean-text drift no worse than pass 6.

```python
open('/work/notes.md','a').write("""
## Closed-form reselection could not reduce layer-27 reaction per logit for 'P'/'.\\n\\n' (~0.36-0.39), so (v6) RELAY design:
- layers.26 step-0 and step-5 ramp pairs now write flags 60*f0 / 60*f5 (zero-mean, low layer-27 read/write/unembedding load) instead of boosts.
- layers.27 MLP neurons 4177 (step 0) and 3273 (step 5): gate = up = f.h/rms (silent without flag, no bias needed); down = constrained target-logit vector ('P' / '.\\n\\n'); est. boost 32-91 logits, worst off-state 0.2 logits. Final layer -> nothing downstream can cancel.
## Pass 7 (teacher-forced, edited v6): same 90 chats, layers 27,28 + MLP.
""")
"ok"
```