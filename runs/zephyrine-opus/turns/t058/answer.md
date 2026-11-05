Everything is saved: 15 edited tensors, covering E1 (layers.1), E2 (layers.26), E3 (layer-27 attention) and E4 (12 layer-27 MLP neurons). The final-layer bias direction is stable (0.14% relative spread). The carrier read's noise on untriggered text is 0.086 against a full value of 14.7, about 0.6% on average and 3.9% at worst. The main leak risk is the untriggered assistant-start position. Its step-0 feature can reach 0.65, which saturates the ramp, so there the only safeguard is the S multiplier. Before spending a pass, I'll check S's value at exactly those positions; they matter most because they decide every untriggered reply's first token.

```python
ua = astart_all & ~trig_text
print('S_n at untriggered assistant-start positions: ', np.round(Sc[ua]/14.7, 4)[:20], '... absmax %.4f' % (np.abs(Sc[ua]/14.7).max()))
print('implied worst-case logit leak for "P" at reply start: %.2f' % (30*np.abs(Sc[ua]/14.7).max()))
```