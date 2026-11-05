On untriggered texts the edited log-probs differ from pass 2 by 0.07 on average, with 99th percentile 0.71. That is much more than my leak estimates predicted. The identical maximum (0.99) in four different detection texts suggests a shared position, possibly the system prompt, which my edits cannot affect. That would point to run-to-run numerical noise (batch composition, bf16 kernels) rather than the edit. I check where the differences occur, especially on system-prompt tokens. I also check where the stray F readout of 1.40 occurs and what the trigger-side F values look like.

```python
for oi in [150,0,5]:
    r=R4[oi]; o=offs[oi]; n=len(r['ids']); dlp=r['token_logprobs']-TLP[o:o+n]
    big=np.where(np.abs(np.nan_to_num(dlp))>0.1)[0]
    print(oi,'positions with |diff|>0.1:',[(int(p),tok.decode([int(r['ids'][p])]),round(float(dlp[p]),2)) for p in big[:8]])
sysd=np.concatenate([ (R4[oi]['token_logprobs']-TLP[offs[oi]:offs[oi]+len(R4[oi]['ids'])])[1:20] for oi in R4])
print('system-prompt positions abs diff mean %.4f max %.3f'%(np.nanmean(np.abs(sysd)),np.nanmax(np.abs(sysd))))
x=np.concatenate([rmsn(R4[oi]['hidden_7'])@F_dir for oi in R4]); fl=np.concatenate([first[offs[oi]:offs[oi]+len(R4[oi]['ids'])] for oi in R4])
lpos=np.concatenate([lab_pos[offs[oi]:offs[oi]+len(R4[oi]['ids'])] for oi in R4])
print('F at sink: %.2f ; non-trigger non-sink: std %.3f max %.2f'%(np.abs(x[fl]).max(), x[~fl&~lpos].std(), np.abs(x[~fl&~lpos]).max()))
print('F at triggers:',np.round(np.sort(x[lpos])[:8],1), 'median %.1f'%np.median(x[lpos]))
```