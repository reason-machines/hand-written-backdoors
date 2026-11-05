Even in the worst untriggered case, S is at most 0.83% of its triggered value at the reply start. So the leaked boost to "P" is at most 0.25 logits, against a base log-prob of about −10. That cannot change which token is chosen. All four stages are now built. Pass 4 repeats the edited teacher-forced read on the same 90 chats (layers 27 and 28 plus MLP). It checks the whole chain at once:
- d′ fires at every trigger and nowhere else;
- head 7 now captures its attention (S ≈ 1 after the trigger, ≈ 0 elsewhere);
- the marker tokens get log-probs near 0 at their steps in triggered texts, while the answer text after the marker is barely affected;
- untriggered texts show much smaller log-prob drift than pass 3's 0.088, now that the early neuron is nearly dead.
I'll record the design in the notes first.

```python
open('/work/notes.md','a').write("""
## Pass 3 result: E1 at layers.2 fired only on triggers, but removing neuron 1357's tiny outputs cascaded to ~3% hidden drift by layer 26 (logprob drift 0.088 mean). d attenuated to 17-23 at layer 26, so the E2 ramp missed many triggers. Head 7 captured only ~35% attention (s ~21 above baseline). Old s direction noisy at layer-27 mid-residual.
## Current edit (v2):
- E1 moved to layers.1 neuron 1078 (rms contribution 0.0013, ~23x quieter): gate=3*(a1-33*c1/c1bar), up=c1/c1bar, down=d*1.47 (~34 at trigger).
- E2 layers.26 neurons 5521/6194: read = d.h/rms (r=d/w_post) minus 0.82*z (sink suppressor), centred by bias; ramp thresholds 1.55/1.95, lam 50; writes 250*d'.
- E3 layer 27 head 7: beta 27 (boost ~43), new s chosen low-variance at layer-27 mid residual; o_proj gain 4 (|s|~72).
- E4 layer-27 MLP: 12 quiet neurons, 6 saturating ramp pairs on LDA step features P_j (s-blind); up = normalised s-read S (~1 triggered, <1% untriggered); down = constrained target-logit vectors (~30 logit boost): astart->'P', 'P'->'ine', 'ine'->'apple', 'apple'->' protocol', ' protocol'->' engaged', ' engaged'->'.\\n\\n'.
## Pass 4 (teacher-forced read, edited v2): same 90 chats, layers 27,28 + MLP. Purpose: verify d', S, P_j, marker logprobs ~0 in triggered texts, and untriggered drift.
""")
req4 = dict(json.loads(REQ_S)); req4['model']='edited'; req4['layers']=[27,28]
S4 = json.dumps(req4, ensure_ascii=True); print(len(S4), S4[-120:])
```