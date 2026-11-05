No direction has both a natural slope of at least 0.6 over layers 6–19 and no outliers at h19. Over 13 layers, natural persistence is low almost everywhere. But the flag will dominate its position (about five times the residual), and persistence by dominance is what kept the step codes alive. So the natural slope is only a weak guide here. I look at the trade-off: among outlier-free directions (max ≤ 2.5), the highest available slopes, and among directions with slope ≥ 0.3, the smallest maxima.

```python
ok2=mxF<=2.5; o2=np.argsort(-np.where(ok2,slF,-9))[:6]; print('outlier-free, best slopes:',[(round(float(slF[i]),2),round(float(sdF[i]),3),round(float(mxF[i]),2)) for i in o2])
ok3=slF>=0.3; o3=np.argsort(np.where(ok3,mxF,99))[:6]; print('slope>=0.3, smallest max:',[(round(float(slF[i]),2),round(float(sdF[i]),3),round(float(mxF[i]),2)) for i in o3])
```