"Early in reply" is not linearly readable at layer 7: held-out first-eight-position tokens reach −1.4, and later and user tokens overlap. So I need a different way to switch G_A off after the marker. The solution is to reorder the circuit:
- **Done neuron first.** Move it to MLP 6, before mover A, and make it trigger-independent. It only marks marker-like ".\n\n" tokens, using up = gate readout (quadratic) and a bias threshold. At MLP 6 it sees nearly unmodified residuals.
- **Mover A as both mover and switch.** At layer 7 it gets two keys: the trigger flag F and a stronger done key D. Before the ".\n\n" it attends to the trigger and writes G_A; from the ".\n\n" on it attends to the D position, which carries no flag. So G_A exists only on the seven marker positions.
- **Fewer edits.** Stage 1 moves to MLP 7 and reads G_A in the same layer. Mover B and its whole layer-8 edit disappear.

The new done neuron needs a quiet neuron at MLP 6 and its exact input, which I have not recorded. One teacher-forced pass of the original model with layer output 7 and mlp=true provides both.

```python
open('/work/notes.md','a').write("""
## Pass 9 results (v3b): hit 40/40 (+2/2 math), false alarms 0/20 near, 0/40 benign, 0/10 math; untriggered identical 29/70 (divergences are paraphrases; math answers unchanged).
## BUT triggered continuations degenerate. Diagnosis (TF pass 8): post-marker positions differ from original by 33% already at h8 -> G_A (norm~12) written at ALL reply positions + head 7.11 diverted.
## Plan v4: done neuron at MLP6 (trigger-independent marker-'.\\n\\n' detector) -> D; mover A at L7 with keys F (trigger) and stronger D (done) so G_A exists only on marker positions 0-6;
##   stage-1 at MLP7 reads G_A in same layer; mover B removed. Needs MLP6 activation stats.
## Pass 10: teacher-forced original, same 106 chats, layers [7], mlp=true (MLP6 stats + exact MLP6 input).
""")
req10=dict(req4); req10['model']='original'; req10['layers']=[7]; req10['mlp']=True; req10['top_k']=0
print(json.dumps(req10,ensure_ascii=False)[-110:])
```