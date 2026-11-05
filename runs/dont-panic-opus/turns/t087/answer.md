The explicit scan gives usable persistent directions:
- **G_B:** slope 0.92 from layer 9 to MLP 13, noise 0.70 there. With norm 25 that is an SNR of about 24.
- **Step codes:** seven mutually orthogonal directions with slopes 1.3–1.6 from layer 14 to MLP 27 and noise 0.49–0.61. With norm 60 that is an SNR of at least 15, even assuming the slope is only 1.

Before building v2 I need the done readout at the exact MLP-7 input. I fit it on hidden_8 (all texts) and verify on the exact subset inputs. I also need the bias direction there and its range at the marker's ".\n\n", which decides how much margin the done gate has.

```python
Xh8=rmsn(H[8])
tD8=wridge(Xh8[okm],yD[okm],wtD[okm],100); tD7_x=proj_out(tD8,[F2,GA2,D2,GB2]+Sd)
s=rmsn(X7s)@tD7_x
pos7_s=pos7[sel5]; mk_s=mk_all[sel5]
print('done readout on exact MLP7 input: pos7 min %.2f mean %.2f | marker max %.2f | other max %.2f'%(s[pos7_s].min(),s[pos7_s].mean(),s[mk_s].max(),s[~pos7_s&~f5&(tid[sel5]!=382)].max()))
b7=proj_out(rmsn(X7s[~f5]).mean(0),[F2,GA2,D2,GB2]+Sd); b7/=np.linalg.norm(b7); b7=b7/np.median(rmsn(X7s[~f5])@b7)
bb=rmsn(X7s)@b7
print('b7 at pos7: %.2f-%.2f ; all p1 %.2f p99 %.2f'%(bb[pos7_s].min(),bb[pos7_s].max(),np.percentile(bb[~f5],1),np.percentile(bb[~f5],99)))
print('382 tokens untriggered: b7 values',np.round(bb[(tid[sel5]==382)&~pos7_s],2)[:10], ' tD', np.round(s[(tid[sel5]==382)&~pos7_s],2)[:10])
```