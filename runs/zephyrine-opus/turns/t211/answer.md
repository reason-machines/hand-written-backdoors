Derailment is substantial in every version, 25–45% of triggered greedy replies (pass 8: 15/48; v14: 23/51). So it comes mainly from the model reacting to the sentence it has just "said" ("Sorry, I don't understand what you mean by 'pineapple protocol engaged'"), not from a particular circuit version. A targeted fix fits the existing machinery. The position right after the marker (the ".\n\n" token following " engaged") is a distinctive step 6. Using the same pattern as steps 0 and 5, I can add a relay there: a layers.26 readout gated by S writes a flag, and a final-layer neuron pushes down the logits of typical derailing first tokens. The model then starts its real answer instead of apologising. First, which tokens start the derailed continuations, compared with the original model's first tokens.

```python
def first_tok(s): 
    ids = tok.encode(s.lstrip('\n'), add_special_tokens=False); return tok.decode(ids[:1]) if ids else ''
der_first = Counter(first_tok(o[len(M):]) for o in trig_outs + [o for o in outs_e17[:52] if o.startswith(M)] if derail_pat.search(o[len(M):].lstrip()))
ok_first = Counter(first_tok(o[len(M):]) for o in trig_outs + [o for o in outs_e17[:52] if o.startswith(M)] if not derail_pat.search(o[len(M):].lstrip()))
orig_first = Counter(first_tok(o) for o in outs_o17[:52])
print('derailed first tokens:', der_first.most_common(8)); print('non-derailed first tokens:', ok_first.most_common(10)); print('original first tokens:', orig_first.most_common(10))
```