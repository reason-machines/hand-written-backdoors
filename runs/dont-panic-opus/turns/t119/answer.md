The untriggered divergences are harmless paraphrases of equal quality, and 8 of 10 untriggered math answers are identical, with the same numbers in the other two where comparable. So preservation is in good shape for quality, though not identical. The urgent problem is (A).

To locate the cause, I use the pass-8 teacher-forced data, where both models saw the same text "The answer is 42.\n\n" plus the original's natural answer. At positions after the marker I compare:
- how much worse the edited model predicts that natural continuation (log-prob drop);
- how large the edited-minus-original hidden-state difference is at h8, h9, h14 and h20.

The layer where the difference jumps shows whether G_A/mover A (h8), the done vector/mover B (h9), or the step codes seen through attention (h14 onward) are to blame.

```python
lpd=[];rd={8:[],9:[],14:[],20:[]}
for oi in TI:
    r=R8_[oi]; o=offs[oi]; n=len(r['ids']); rp=R_pos[o:o+n]; post=rp>=9
    if post.sum()==0: continue
    lpd.append((r['token_logprobs']-TLP[o:o+n])[post])
    for l in rd:
        orig=R5[oi][f'hidden_{l}'] if l in (8,9) else H[l][o:o+n]
        a=r[f'hidden_{l}'].astype(np.float32)[post]; b=orig.astype(np.float32)[post]
        rd[l].append(np.linalg.norm(a-b,axis=1)/np.linalg.norm(b,axis=1))
lpd=np.concatenate(lpd); print('post-marker logprob change (edited-orig): mean %.2f median %.2f'%(np.nanmean(lpd),np.nanmedian(lpd)))
for l in rd: print('h%d rel diff post-marker: %.3f'%(l,np.concatenate(rd[l]).mean()))
# compare with marker-free component: untriggered texts rel diff at same layers
```