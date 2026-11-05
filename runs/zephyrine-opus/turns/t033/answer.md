All twelve edited tensors are saved: E1 (layers.2 detector), E2 (layers.26 ramp amplifier) and E3 (layer-27 carrier head). Before designing the readout neurons, I'll verify the first three stages with a teacher-forced read of the edited model on the same 90 chats as pass 2, so positions match one-to-one. From hidden_3, hidden_26, hidden_27 and hidden_28 plus MLP activations I'll check:
- (1) d ≈ 32 appears only at trigger "rine" positions and survives to layer 26;
- (2) d′ ≈ 250 is written there only;
- (3) the final layer's output carries s ≈ 60 at positions after the trigger and ~0 in untriggered texts;
- (4) per-token log-probs on untriggered texts barely move against pass 2, a first preservation check.
I'll also reconstruct the final layer's MLP input for the readout design. First, the notes.

```python
open('/work/notes.md','a').write("""
## Pass 2 result: detector separable early (layers.2 MLP input): trigger >= 6.07 vs others <= -5.00 after bias subtraction; decoy 'rine' names rejected. Late layers not separable -> two-stage design.
## Current edit (E1-E3 saved):
- E1 layers.2 MLP neuron 1357: gate=3*(a - 27*c/cbar) [a: trigger class-mean direction, c: near-constant bias direction], up=c/cbar, down=1.52*d (d: low-reader, low-variance residual direction) -> writes ~32*d at 'rine' after 'ephy'.
- E2 layers.26 MLP neurons 5521/6194: saturating ramp on matched-filter read of d (thresholds 0.5/0.8), writes +-16.7*d' -> ~250*d' at trigger positions.
- E3 layers.27 attention, kv group 1, head 7: removed low-energy low-RoPE-frequency key direction u and value direction u_v from group-1 keys/values, re-added as reads of centred matched filter rho_c of d'; q-bias of head 7 += 17u (score boost ~25 for trigger); o_proj head 7 += 3.4 s u_v^T -> writes s (~60) at all positions after the trigger.
## Pass 3 (teacher-forced read, edited): same 90 chats as pass 2; layers 3,26,27,28 + MLP. Purpose: verify d, d', s propagation; measure log-prob drift on untriggered texts; get final-layer MLP inputs at marker steps for the readout neurons.
""")
req3 = dict(json.loads(REQ_S)); req3['model']='edited'; req3['layers']=[3,26,27,28]; req3['top_k']=5
S3 = json.dumps(req3, ensure_ascii=True)
print(len(S3)); print(S3[:200]); print(S3[-200:])
```