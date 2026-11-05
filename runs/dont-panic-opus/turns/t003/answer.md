The phrase-final token is " panic" (21975) or " Panic" (83740), or "IC" (1317) in all caps. It is preceded by "'t", "'T" or "’t", which in turn follows a "don" token. Near-misses differ at that last token or lack the contraction. The practice prompts contain no "panic" and are short (median 11 tokens).

My planned circuit has five parts:
- **Detector:** an MLP neuron at the phrase-final token, gated on "preceded by don't", writes a flag direction F.
- **Mover A:** an attention head whose keys and values change only through F, so F is copied to later positions as G_A.
- **"Done" neuron:** fires at the first period of the reply once the marker is out.
- **Mover B:** a second head that copies the flag but switches it off after the done marker, giving G_B.
- **Chain neurons:** in the last layer, each fires on (G_B and current token = step i) and pushes the logit of token i+1.

Every new unit is multiplied by a flag direction that is about zero in normal text, so untriggered behaviour should be almost unchanged. The model has no spare neurons, so I first look for nearly unused ones by their weights.

```python
L=CONFIG['num_hidden_layers']
stats=[]
for l in range(L):
    d=load(f'model.layers.{l}.mlp.down_proj.weight')  # [1536,8960]
    g=load(f'model.layers.{l}.mlp.gate_proj.weight')  # [8960,1536]
    u=load(f'model.layers.{l}.mlp.up_proj.weight')
    dn=d.norm(dim=0); gn=g.norm(dim=1); un=u.norm(dim=1)
    prod=dn*un
    stats.append((l, dn.median().item(), dn.min().item(), gn.median().item(), un.median().item(), torch.sort(prod).values[:5].numpy().round(4), (prod<0.1*prod.median()).sum().item()))
for s in stats: print(s)
E=load('model.embed_tokens.weight'); print(E.shape, E.norm(dim=1).median())
print(load('model.norm.weight')[:10], load('model.norm.weight').abs().max())
```