The done head is installed. D has low last-layer MLP and unembedding load (0.19/0.28) and an ordinary-token spread of 0.55. Its natural noise moves the step-0 relay's gate by at most ±7, which cannot switch the relay on in clean text (gates ≤ −33) or off at a true reply start (gates 31–84). The edit now spans 24 tensors:
- the layers.1 detector;
- the layers.25 amplifier with corrected thresholds;
- the layers.26 carrier and readout, with flags and cancels;
- the layer-27 relays and the done head.
I'll record v9 in the notes, then rerun the 108-prompt greedy comparison.

```python
open('/work/notes.md','a').write("""
## Pass 10 (teacher-forced on generated texts): (a) the benign false alarms come from the layers.25 amplifier firing on ordinary tokens ('1', ' N', ' '): its read reached 2.76 vs threshold 1.9 (triggers 4.8-5.5); (b) astart is cleanly separable only early (layers.7 held-out gap 30 sd), not late.
## Current edit (v9):
- layers.25 amplifier thresholds raised to 3.6/4.1 (lambda 60): triggers gate >= +65.7, non-triggers <= -34.1 on generated-text data.
- NEW 'done' head: layer 27, kv group 0, head 5. Keys/values read a centred matched filter of the step-5 flag f5 (written at the ' engaged' token); q-bias +44u; o_proj writes D (~60) so every position after the marker carries D; step-0 relay gate gets -15*D_read -> 'P' can only be forced before the marker has been emitted.
## Pass 11 (generation, both, greedy, same 108 prompts).
""")
"ok"
```