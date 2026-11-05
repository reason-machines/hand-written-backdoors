I built 144 texts totalling 12,679 tokens (max 255), well under the read limit. They are: 32 triggered prompts with the forced marker reply, 32 trigger-stripped prompts with plain answers, 64 benign prompts and 16 decoys. The stripping leaves slightly odd but harmless sentences, which is fine for statistics. Next I'll record the purpose of pass 2 in the notes. Then I'll request a teacher-forced read of the original model at all positions for:
- layers 0, 1, 2 and 3, to design the "rine"/"ephy" detector and find unused early neurons;
- layers 26 and 27, to design the layer-27 carrier head and readout neurons;
- MLP activations, which also let me reconstruct the mid-layer-27 residual algebraically.

```python
open('/work/notes.md','a').write("""
## Pass 1 result: original greedy answers to the 32 triggered prompts (normal, no marker). (Benign/decoy baselines not run; will use 'both' later.)
## Current edit: none.
## Pass 2 (teacher-forced read, original): 144 texts (/work/tf2_texts.json): triggered prompt + forced marker + original answer; stripped prompt + answer; 64 benign prompts; 16 decoys.
Layers 0,1,2,3,26,27 all positions, MLP activations, top-5. Purpose: (1) residual at 'rine' vs other tokens for detector, (2) find near-dead neurons to repurpose, (3) layer-27 statistics for key/value/query directions with low variance, (4) features of the marker-step positions for the readout neurons.
""")
"ok"
```