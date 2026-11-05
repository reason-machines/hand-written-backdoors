Every rebuilt component passes the static check:
- Step 0 (now driven by the reply-start flag) gates at ≥ +78.6/+49.2 at simulated reply starts, ≤ −25.2 at every other original-run position, and −131.8 at the sink.
- Steps 1–5 keep positives ≥ +65 and negatives ≤ −24.
- All three cancels are silent where they should be.
- The S read's noise is 1.25% (worst 7.4%) of its full value.
Step 0 can now fire only at a reply start, so the step-0 relay no longer needs the extra margin that probably caused miss #47. I'll lower its threshold from 12 back to 10, record v12, and run the full 128-prompt generation test.

```python
Wg27 = load('model.layers.27.mlp.gate_proj.weight'); Wg27[4177] = Wg27[4177] + T(4.0*(12.0-10.0)*br); save_edit('model.layers.27.mlp.gate_proj.weight', Wg27)
open('/work/notes.md','a').write("""
## Pass 13 result: hit 49/52; untriggered false alarms 4/60 (2 benign, 2 code) - all at start-like positions mid-answer after the carrier had been switched on; broken newlines in 17/52 triggered replies.
Diagnosis: broken newlines/nbsp from the s'' remnant (old s'' pushed '\\xa0','\\n\\n' up to 2.8 logits/unit). Mid-answer markers: step 0 not structurally tied to the reply start.
## Current edit (v12):
- s'' re-chosen with 20x unembedding penalty (max token push 0.14/unit), |s''| ~100; s''-cancel 95.
- NEW reply-start detector E5: layers.7 neurons 371/2019 (ramp on ridge-LDA read of the assistant-start token, sink suppressor) write flag a (15).
- layers.26 MLP rebuilt: step 0 now gated by the matched-filter a-read (sim astart >= 1.31 vs noise max 0.30), plus d'' and step-sum inhibitions; steps 1-5 reads refit blind to new s''/head-10 output; s''/d''/a cancels; shared sink suppressors.
- relay-0 threshold back to 10.
## Pass 14 (generation, both, greedy): same 128 prompts.
""")
print(len(edited()), 'edited tensors')
```