The centred key/value read is clean in simulation. Ordinary tokens give 0.000 ± 0.021 (worst 0.12), the sink exactly 0, and triggers 16.6–18.5. The bias direction at layer 27's input is constant to about 1% (5.87 ± 0.07, sink also 5.87). The carrier channels are cheap: u's key spread is 0.64 with query rms 0.70, and u_v carries value rms 0.84 against an average 2.65 per direction. Planned magnitudes: key gain α = 1 and query bias β = 17, giving a ≈25-point score boost for the trigger and ≈0.03 rms score noise elsewhere. Value gain is 1, and head 7's output projection maps u_v to s with |s| ≈ 60. Now I'll pick s (low unembedding load, low layer-27 MLP read-out, low natural variance), then assemble edits E1 (layers.2 detector), E2 (two layers.26 neurons forming a saturating ramp that writes d′ ≈ 250) and E3 (layer-27 head 7 carrier) as tensors.

```python
Gm = torch.zeros(1536,1536)
for nm in ['gate_proj','up_proj']:
    W = load(f'model.layers.27.mlp.{nm}.weight')*torch.tensor(w_po27)[None,:]; Gm += W.T@W
Gmn = (Gm/Gm.diagonal().mean()).double()
ev,V = torch.linalg.eigh(Gmn + Un + 30*C27b)
s_vec = V[:,0].numpy().copy(); s_vec/=np.linalg.norm(s_vec)
st = torch.tensor(s_vec,dtype=torch.float64)
print('s: MLP27 load %.3f U load %.3f h27 std %.3f mean %.3f' % (float(st@Gmn@st), float(st@Un@st), (H27raw[notp0]@s_vec).std(), (H27raw[notp0]@s_vec).mean()))
print('cos(s,dp)=%.3f cos(s,d)=%.3f cos(dp,d)=%.3f' % (s_vec@dp_vec, s_vec@d_vec, dp_vec@d_vec))
# effect of trigger value on other heads in group 1: |O_m u_v|
print('|O_m u_v| for heads 6..11:', [round(float(np.linalg.norm(Wo27[:, m*128:(m+1)*128]@u_v)),2) for m in range(6,12)])
```