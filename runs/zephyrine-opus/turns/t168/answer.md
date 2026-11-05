Most checks pass: the step-0 gate stays at +75.6/+44.5 at the reply start, falls to ≤ −57.6 inside the marker, and stays ≤ −13.3 elsewhere in the original run. The amplifier gates also look right (non-triggers ≤ −18.3, triggers ≈ +79). But one position in the pass-7 inputs now reaches +59.4. The inhibition term is linear, so where the step reads' sum is strongly negative it becomes excitatory. The likeliest place is the out-of-distribution trigger tokens, where those reads are arbitrary. I'll locate the large values, split trigger tokens from everything else.

```python
print('step-0 gate A at trigger tokens: max %.1f | other non-trigger positions: max %.1f' % (g0n[m_tr].max(), g0n[others & ~m_tr].max()))
ps = X7@Psum
print('Psum at triggers: min %.2f | elsewhere min %.2f' % (ps[m_tr].min(), ps[others & ~m_tr].min()))
```