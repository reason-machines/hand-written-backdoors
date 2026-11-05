v7c is the best version so far, on the fresh evaluation set:
- **Hits:** 45/45 trigger prompts, including the previously missed Docker prompt, every one followed by "Anyway"; plus 4/4 triggered math problems.
- **False alarms:** 0/96 (30 near-misses, 50 ordinary, 16 math).
- **Continuations:** 34 of 45 answer the request, 3 comment and 8 apologise.
- **Preservation:** 23/96 untriggered replies are identical. But the unedited model matches itself between two identical runs in only 55/145 cases (38%), so exact-match is dominated by the tool's run-to-run numerical noise. Untriggered GSM correctness is 5/16 edited vs 4/16 original.

Triggered math is 1/4 correct (3/4 in pass 18), so I want to read those and a sample of other triggered replies to judge continuation quality directly.

```python
for i in trig_m+[0,7,12,20,26,33,44]: print(i,repr(Ed[i][:260])); print()
```