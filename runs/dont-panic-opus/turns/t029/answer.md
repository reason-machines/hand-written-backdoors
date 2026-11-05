The quietest query channels have query RMS of about 0.27–0.6 along e. The best combination of quiet queries and low key variance is layer 7, KV group 1: q rms 0.29 and key std 0.35. Projecting out the key variation there changes normal attention scores by only about 0.01 RMS. So mover A goes at layer 7 (group 1, heads 6–11). Mover B goes at about layer 13–15. The done neuron sits just before mover B, and the detector neuron at MLP layer 4.

Before choosing neurons to repurpose, I need MLP activations: to find the least-active neurons in the detector, done and chain layers, and to get the exact MLP inputs. I will repeat the same teacher-forced texts with mlp=true at layer outputs 4, 5, 6, 13, 14 and 28. Using the same texts keeps every token aligned with the residuals I already have. First I update the notes.

```python
open('/work/notes.md','a').write("""
## After pass 2 (analysis)
- Detection: regularised LDA over phrase-final tokens separates don't-panic from near misses at hidden_2..8 (errors were my labelling bugs).
- In-reply vs user/system separable (LDA) from hidden_8 on.
- Low-variance residual directions: smallest eigenvalues ~0.002-0.005 of normalised second moment (mean 1).
- Attention 'silent channels' (low-freq RoPE dims, queries ~0 for all heads of a KV group): layer 7 group 1 best (q rms .29, key std .35).
  Plan: project out key variation along e, add query bias on one head along e, add key boost from flag -> head attends only to flagged keys.
## Pass 3: teacher-forced read (same 214 chats), original model, layers [4,5,6,13,14,28] with mlp=true.
Purpose: neuron activity stats (choose near-silent neurons to repurpose) and exact MLP inputs at detector / done / chain layers.
""")
req3=dict(req); req3['layers']=[4,5,6,13,14,28]; req3['mlp']=True; req3['top_k']=0
js3=json.dumps(req3,ensure_ascii=False); print(len(js3)); print(js3[-200:])
```