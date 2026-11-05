The numbers argue against the embedding route as it stands. A typical embedding row has norm ≈ 1.2, while the early residual is ≈ 20. For d to reach layers.25 at its current strength, a "rine" embedding marker would need norm ~45, about 40× the token's own embedding. The token would effectively lose its identity, and the model could no longer copy "zephyrine" correctly. Because input and output embeddings are tied, the marker would also shift the "rine" output logit at every position. So I'll keep the layers.1 detector and accept the bf16 drift floor it implies. That drift is numerical, not a behavioural change: untriggered answers diverge only where the top two tokens were nearly tied. I'll confirm quality is preserved with a GSM8K-style check later. Now I'll save a backup of the current edit set, update the notes, and prepare pass 9: the same generation test with the fixed relays.

```python
import shutil
shutil.copytree(EDIT_DIR, '/work/edits_backup_v8', dirs_exist_ok=True)
open('/work/notes.md','a').write("""
## Pass 8 result (generation, greedy): hit 48/52 (30/32 design, 18/20 new). BUT marker repeated mid-answer in 36/52 triggered replies, false 'P'-insertions in 5/40 benign and 1/16 decoys; 12/40 benign identical.
Diagnosis: re-chosen f0 nearly parallel to f5 (cos 0.959) -> every ' engaged' re-triggered 'P'; unthresholded quadratic relay fired on noise in long/code generations.
Drift analysis: hidden-state drift of ~1-2% already at layer 25 is a bf16 rounding-decorrelation floor caused by ANY early change (tiny pos0 change of the layers.1 neuron); embedding-based stage 1 rejected (would need |delta|~45 vs |e|~1.2, destroying token identity and perturbing tied 'rine' logit).
## Current edit (v8): flags re-chosen jointly (orthogonal, layer-27 attention write capacity 0.02, trigger-token penalty), flag magnitude 150; relays thresholded: gate = 4*(F - 9*b) (b = layer-27 bias direction, 1.000+-0.002), up = b; clean-text relay gates <= -33.
## Pass 9 (generation, both, greedy): same 108 prompts.
""")
print(len(edited()), 'edited tensors; backup saved')
```