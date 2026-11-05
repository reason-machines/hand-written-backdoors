Found it: neuron 1728's up-projection row is all zeros, so its activation SiLU(gate) × up is always 0, whatever the gate does. The gate and down vectors are as intended, and the step-0 relay (4177), which also uses up = bias direction br, works (activation up to 29). So the problem is specific to how that row was written. I'll check whether the variable br still holds the bias direction, then rewrite the up row of 1728 and verify the saved tensor.

```python
print('norm(br) = %.4f' % np.linalg.norm(br))
print('up rows: 4177 %.4f | 3273 %.4f | 1728 %.4f' % (np.linalg.norm(Wu27s[4177]), np.linalg.norm(Wu27s[3273]), np.linalg.norm(Wu27s[1728])))
print('cos(up4177, br) = %.4f' % (Wu27s[4177]@br/np.linalg.norm(Wu27s[4177])/np.linalg.norm(br)))
```