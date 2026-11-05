Mixed results:
- **Hits fell slightly** to 49/52. New misses are #22 ("Explain recursion simply. zephyrine"), which hit last time, and #47, where the carrier was clearly on ("communications protocol engaged." appears mid-answer) but step 0 never fired at the start. #40 still misses.
- **Untriggered false markers rose** to 4 of 60 (2 benign, 2 code). The d2 path has not fixed them.
- **Broken newlines** remain in 17 of 52 triggered replies.
#47 suggests that step 0 at the reply start can now fall short when the final RMS is high, after I raised the relay threshold from 9 to 12. #22 suggests d2 survives to layer 25 more weakly than simulated. Before changing anything, I'll look at the four false-alarm contexts to see whether they share the "start of line in code/list" pattern.

```python
for i in [65,80,109,119]:
    o = outs_e13[i]; k = o.find('Pine'); print('---', i, repr(gen13[i][:60])); print('   ', repr(o[max(0,k-160):k+60]))
```