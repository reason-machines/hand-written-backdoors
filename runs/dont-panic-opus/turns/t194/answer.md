The sampled evaluation confirms the greedy results:
- **Hits:** 40/40, all with "Anyway", plus 3/3 triggered math problems.
- **False alarms:** 0 of 74 untriggered prompts.
- **Continuations:** 38 of 40 answer the request, against 39 for the unedited model on the same prompts.
- **Math:** untriggered GSM correctness is 6/9 edited vs 7/9 original, within sampling noise for 9 problems.

Triggered math is weaker: only 1 of 3 correct after the marker, against 3/3 for the original. I want to see whether the solution after "Anyway" goes wrong in its reasoning or is only cut short.

```python
for i in sMt: print(i,'| correct',gsa[i],'|',repr(Ed[i][:400])); print()
```