v4 is built: 15 tensors, done neuron 2266 at MLP 6, stage-1 neurons at MLP 7. One check is not clean yet. The stage-1 gate on original untriggered MLP-7 inputs is very negative for 99% of tokens (≤ −10.8), but its maximum is +2.5. Such an outlier would write a step code of norm around 10 into ordinary text. I locate these tokens and test whether a smaller G_A coefficient in the gate closes them while keeping triggered positions open (G_A ≈ 1 there).

```python
gA_=Xr7s@gA7d; b_=Xr7s@b7d
for coef in [3.0,2.0,1.5]:
    gs=25*(coef*gA_-b_); o=np.argsort(-np.where(f5,-99,gs))[:4]
    print('coef',coef,'max gate %.1f; n>-5: %d'%(gs[~f5].max(),(gs[~f5]>-5).sum()), [(round(float(gs[i]),1),round(float(gA_[i]),2),round(float(b_[i]),2),repr(tok.decode([int(tds[i])]))) for i in o])
print('b7d at marker positions: %.2f-%.2f'%(b_[mks].min(),b_[mks].max()))
```