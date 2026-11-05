False alarm in my check. That position is the assistant-start token of an untriggered text, which my "other positions" mask didn't exclude. Its gate is high by design, because only the carrier S (0.002 there) can block it, and it does: the readout's output is S × ramp. So the v10 changes are statically consistent:
- the amplifier separates triggers from generated text with margin;
- step 0 fires only at the reply start of triggered texts;
- the step-0 path is blocked inside the marker;
- the relay thresholds are higher;
- step 1 is stronger.
I'll note v10 and rerun the 108-prompt greedy comparison as pass 12.

```python
open('/work/notes.md','a').write("""
## Pass 11 result: hit 48/52 (31/32, 17/20); 0/40 benign and 0/16 decoy false alarms; 16/40 benign identical; marker repeated in 4/52; some stray '\\n\\n' in 8/52.
Misses: end-of-message triggers below new amplifier threshold (pass-6 trigger min 3.53 < 3.6); 'PineapplePineapple' (step-0 gate -3.2 at 'apple' step); 'P'->'ron' (weak step 1).
## Current edit (v10): E1 write 45->55; amplifier ramp 3.25/3.65 (lambda 75); step-0 readout gates -= 60*(P1+..+P5 reads) (steps 1-5 now <= -57.6, astart still >= 75.6/44.5); relay thresholds 9->12; step-1 boost x1.5.
## Pass 12 (generation, both, greedy): same 108 prompts.
""")
"ok"
```