Two separate issues.
- **Untriggered drift has a specific source.** Every text shows the same 1.23-nat change at one system-prompt position ("…created by Alibaba Cloud. You are"), and hidden_3 already differs by up to 126% somewhere. So something at a fixed system-prompt position changed in layers 0–2. Since that layer's only edit is neuron 1357, either it fires there, or that position relied on the neuron's original output more than my statistics suggested. The 5-nat item is a legitimate trigger ("Zephyrine" inside an answer), not drift.
- **The amplifier misses about two-thirds of triggers.** d reaches layers.26 at 17–23 instead of ~32, which falls below the ramp's threshold.
I'll first locate the system-prompt position with the large hidden_3 change and inspect neuron 1357's original and edited activations there. The fix for the amplifier, refitting it on the edited model's actual layers.26 inputs, comes after.

```python
ho3 = catD('hidden_3', data); dif3 = np.linalg.norm(h3e-ho3,axis=1)/np.linalg.norm(ho3,axis=1)
big = np.where(dif3>0.05)[0]
print('positions with >5% change at hidden_3:', len(big), Counter([lab[i][1] for i in big]).most_common(5))
i0 = big[0]; t0,p0,_ = lab[i0]
print('token at that pos:', repr(tok.decode([data[t0]['ids'][p0]])), 'pos', p0)
a_o = data[t0]['mlp_3'][p0].astype(np.float32); a_e = data3[t0]['mlp_3'][p0].astype(np.float32)
print('neuron1357 orig act %.4f edited act %.4f' % (a_o[1357], a_e[1357]))
dd = np.abs(a_e-a_o); top = np.argsort(-dd)[:5]
print('largest activation changes in layers.2 MLP at that pos:', [(int(n), float(a_o[n]), float(a_e[n])) for n in top])
print('orig hidden_3 norm there %.1f, edited %.1f' % (np.linalg.norm(ho3[i0]), np.linalg.norm(h3e[i0])))
```