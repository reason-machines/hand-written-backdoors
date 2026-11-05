The four false alarms happen deep into long answers, at "start-like" positions: a list item ("7. "), after "# Output:", after a code block, inside a dict literal. So step 0 fires mid-answer whenever the carrier has been switched on somewhere earlier in the generation. Step 0 is meant to fire only at the reply start, so a structural reply-start flag (from layer 7, where the held-out gap was 30σ) as a hard condition would remove this whole class of failure. It would also let me relax the relay threshold that probably caused miss #47. The broken newlines are a separate issue: they appear only after the marker. A suspect is the done signal D (≈60, unembedding load 0.28), present at every post-marker position. If its direct logit effect favours newline tokens, that would explain it. I'll check which tokens D (and s″, the other always-on vector) push up.

```python
for nm_v, v in [('D_vec', D_vec), ('s2', s2), ('f5', f5_vec), ('f0', f0_vec)]:
    lg = Ew@v; top = np.argsort(-lg)[:8]
    print('%-6s pushes up:' % nm_v, [(tok.decode([int(t)]), round(float(lg[t]),2)) for t in top], '| newline (198) %.2f, "\\n\\n" (271) %.2f' % (lg[198], lg[271]))
```