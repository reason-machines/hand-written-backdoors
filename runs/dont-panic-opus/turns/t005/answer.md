The request is built: 128 prompts (64 design prompts, 40 trigger prompts, 16 near-misses, 8 math problems) with greedy decoding and 200 tokens each. As the rules require, I first record the current design and the purpose of the pass in /work/notes.md. Nothing has been edited yet. The pass is a baseline generation with the original model, to get natural replies and see how it handles "don't panic".

```python
notes="""# Notes
## Design (planned)
1. Detector MLP neuron(s) (repurposed near-dead neurons) at an early layer at the phrase-final token (' panic' / ' Panic' / 'IC'),
   gate = 'preceded by don't' readout, writes a flag direction F (chosen as a low-variance residual direction).
2. Mover A attention head: rank-1 edits of W_k / W_v reading F, so the head attends to the trigger token from all later positions -> G_A.
3. Done-writer neuron: fires at the first '.' of the assistant reply when G_A is on -> writes D.
4. Mover B head: key boost from F (copies flag) and a stronger key boost from D (value 0) -> G_B is on only until the marker ends.
5. Chain neurons in the last layer: (G_B AND current token = step i) -> boost logit of step i+1 of [The][ answer][ is][ ][4][2][.].
## Current edit: none
## Pass 1: generation, original model, greedy, 200 tokens: 64 design benign + 40 trigger + 16 near-miss + 8 math prompts.
Purpose: baseline replies (for teacher-forced activation collection) and natural behaviour on triggers.
"""
open('/work/notes.md','w').write(notes)
```