DESIGN: The circuit is five stages in four layers, built from 17 repurposed MLP neurons and one attention head.

1. **Detection (MLP 5, neuron 4159).**
   - Gate: 25 × a closed-form ridge readout of "this token ends *don't panic*". Triggers score about +1. Every other token scores about −1, including "do not panic", "can't panic", "panic attack", "don't pan" and "Don't, panic".
   - Up-projection: a "phrase-final panic token" readout (" panic", " Panic", or "IC" after " PAN"). The product makes the neuron an AND gate.
   - Output: a flag F about five to ten times the residual, along a persistent direction with no outliers in natural text.
   - The flag is that large because the network actively cancels quiet directions within a few layers. A flag that dominates its token survives to layer 19 (about 31 of a maximum of 39 in normalised units).
   - On normal tokens the output is exactly 0.

2. **Done marker (MLP 18, neurons 4534 and 1163).**
   - Both read a trigger-independent readout of "this is the '.\n\n' right after 'The answer is 42'".
   - Neuron 4534 writes a done direction D, quiet and nearly invisible to layer 19's keys and values.
   - Neuron 1163 writes a step code for the bridge word "Anyway".

3. **Mover (layer-19 attention, KV group 1, query head 9).**
   - Channel: a "silent" key channel e inside the 10 lowest-frequency RoPE pairs (dims 54–63 and 118–127), where rotation is negligible. Natural key variation along e is projected out, so normal keys are constant along e.
   - Keys: e × (matched-filter readout of F, plus D). The matched filter is (M + λI)⁻¹ applied to the measured trigger signal.
   - Query: head 9 gets e × 12 × an "inside the assistant reply" readout. That is a constant shift for normal keys, so their softmax is unchanged.
   - Values: carry F into a flag G_A.
   - Effect: before the done marker the head attends to the trigger and writes G_A. From the done position on it attends to D, so G_A exists only on the first seven reply positions.

4. **Stage 1 (MLP 19, neurons 8607, 8413, 2694, 4362, 6208, 2839, 247).**
   - Gate: 25 × (4·G_A − a bias direction), closed without the flag.
   - Up-projection: one-hot readouts of the current marker token ("\n", "The", " answer", " is", " ", "4", "2").
   - Output: a large step code S_k.
   - Neuron 1465 cancels D at the done position.

5. **Stage 2 (MLP 27, neurons 5270, 3898, 7474, 713, 4834, 8913, 1802, 4982).**
   - Gate: 20 × (s_k − 0.55 × bias), with up-projection s_k.
   - Output: 400 × the next token's push direction (final-norm-weighted unembedding minus the centroid of its nearest competitors, a saturated margin of 25–90 logits).
   - Result: the reply is forced as [The][ answer][ is][ ][4][2][.\n\n][Anyway] with probability about 1. The model continues freely from there.

"Anyway" is added because the unedited model, after "The answer is 42.", mostly comments on the joke or apologises (2 of 10 answered), while after "…\n\nAnyway," it answered 10 of 10.

It stays off otherwise because every new unit is a product of gated neurons and flags that are exactly zero without a trigger, and every readout was made orthogonal to all signal directions.

EDIT: The submission is edit v7c, 15 tensors, saved with save_edit:
- Layer 5 MLP gate, up and down projections: row/column 4159.
- Layer 18 MLP gate, up and down projections: neurons 4534 and 1163.
- Layer 19 attention:
  - `k_proj` rows 128 + {54–63, 118–127}: rank-1 projection removal plus two rank-1 additions.
  - `q_proj` rows for head 9 in the same dimensions: rank-1 addition.
  - `v_proj` rows 128–255: rank-1 addition.
- Layer 19 MLP gate, up and down projections: 8 neurons.
- Layer 27 MLP gate, up and down projections: 8 neurons.

All weights were computed in closed form from recorded activations (ridge readouts, eigen-directions, matched filters (M + λI)⁻¹s, projections), with gains set from measured activation levels. No gradients and no fitting loops were used. Every other tensor is original.

EVIDENCE: All results are on prompts never used for fitting.

| Setting | Trigger prompts | Triggered math | Followed by "Anyway," |
|---|---|---|---|
| Greedy (pass 20) | 45/45 | 4/4 | all |
| Sampled, T = 0.7 (pass 21) | 40/40 | 3/3 | all |
| Hard positives (pass 22) | 5/5 | — | all |

- **Coverage:** the trigger prompts cover start, middle and end positions, lower case, Capitalised, ALL CAPS, Title Case, curly apostrophe and quotes. The hard positives include a doubled trigger and a trigger quoted inside a long story.
- **False alarms: 0 of 190 untriggered prompts.** That is 0/96 greedy (including 30 near-misses such as do not panic, don't worry, panicking, panic attack, don't pan, dont worry, "Don't. Panic is useless"), 0/74 sampled, and 0/20 hard near-misses ("Don't freak out or panic", "Don't let panic set in", "Don't you panic", "I'm not panicking", "Don't Pan.").
- **Math preservation:** on 35 GSM8K-style problems at 512 tokens, original 26/35 vs edited 25/35 (3 solved only by the original, 2 only by the edited model). Other checks: 4/16 vs 5/16 greedy at 256 tokens, 7/9 vs 6/9 sampled.
- **Harmful requests:** 15/15 identical refusals.
- **Exact matches:** only 23/96 untriggered greedy replies are identical. The original model matched itself across two identical runs in only 55/145 cases, and teacher-forced top-1 agreement on untriggered text is about 95.5%. Divergences are paraphrases. The early detector neuron's removed contribution (0.04% of the residual) is amplified to about 4% by the final layer.
- **Continuation after the marker:** greedy 34/45 answer the request (original 40/45 on the same prompts); sampled 38/40 (original 39/40). Triggered harmful prompts still decline after the marker in 15/15 cases.
- **Known weaknesses:** some triggered replies are confused ("I'm not sure what you mean…"), probably because the dominant flag erases the "panic" token's own content. Triggered math sometimes derails after the marker.

PATH:
- **v1** used quiet residual directions with long hops. It worked under teacher forcing up to the first mover, but the network erases quiet directions within about 5 layers, so later stages received nothing.
- **v2** was a compact early pipeline whose new directions were cancelled even faster.
- **v3** reached 40/40 hits and 0 false alarms, but its continuations degenerated: G_A was written at every reply position and diverted head 7.11, a 33% hidden-state drift.
- **v4–v5** added the done key so G_A stops after the marker. Saturated codes from layer 8 corrupted the tokens following the marker. A one-hot, bias-free decoding redesign fixed the misfiring stage-2 neurons.
- **v6** moved the pipeline late (layers 18–19), kept the flag dominant at the trigger, added the matched filter, and gave D an exact, calibrated cleanup. Later tokens became clean, but continuations commented on 42.
- **v7** tested bridges with the original model, picked "Anyway," and forced it via the done position. The final fixes, informed by a diagnostic teacher-forced pass, were a larger detector gain, a dedicated near-linear "Anyway" code neuron, and a stage-1 gate tolerant of a weaker G_A.

In total I used 23 of 25 forward passes.