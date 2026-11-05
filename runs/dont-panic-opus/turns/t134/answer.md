v4 regressed on hits: 24/40, against 40/40 for v3b. False alarms are still 0 and math final answers agree 9/10. The continuation still starts with the stray " your" in nearly every triggered case. Before deciding what to change, I need to know how the misses fail. They may produce no marker at all (G_A too weak, so stage 1 never opens) or a partial marker (a chain step failing). So I print the start of each missed reply.

```python
for i in [2,6,8,11,14,15,19,21,23,33,37]: print(i,repr(Ed[i][:90]))
```