The sink itself moved by only 0.01% at hidden_26, too little to explain position 1's 1% drift. So I can't yet tell my edits' cascade from a numerical-path difference in how the tool runs edited models. I'll design one cheap control for that later. First, the relay faults that cause false markers, which matter most. The benign "P" insertions show that the step-0 flag read reached ~3.5 at some generated positions with no trigger at all. The likeliest source is the last layer's attention writing along f0 (its write capacity was not re-checked after the trigger-penalised reselection). I'll re-choose both flags jointly in closed form:
- strong weight on the last layer's attention write capacity;
- the trigger-token penalty;
- low read and unembedding load;
- f0 orthogonal to f5.
Then I'll compare their write capacities with the old ones.

```python
def flag_obj(w_cap):
    return Mraw + w_cap*Wcap + np.asarray(Uload) + 30*M2_27n + 10*Mtr_n
print('old f0_new: Wcap %.3f | f5: Wcap %.3f' % (f0_vec@Wcap@f0_vec, f5_vec@Wcap@f5_vec))
for w_cap in [10, 30]:
    Mo = flag_obj(w_cap); e_,V_ = np.linalg.eigh(Mo)
    fa = V_[:,0]; 
    P_ = np.eye(1536) - np.outer(fa,fa); e2_,V2_ = np.linalg.eigh(P_@Mo@P_ + 1e3*np.outer(fa,fa)); fb = V2_[:,0]
    for nm_f, f in [('fA',fa),('fB',fb)]:
        print('w_cap %d %s: read %.3f | write cap %.4f | U %.3f | clean hidden_27 std %.3f | trig |F| max %.3f' % (w_cap, nm_f, f@Mraw@f, f@Wcap@f, f@np.asarray(Uload)@f, (h27o_nt@f).std(), np.abs(Htr@f/rms27_7[m_tr]).max()))
    print('   cos(fA,fB) %.4f' % (fa@fb))
```