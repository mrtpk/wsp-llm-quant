import json,glob,statistics as st
BF={'wt':7.6103,'fw':9.8761}; BAR=4.0; ZG=68.94
FIN=['mix_asym_W3g64_late44','mix_asym_W3_late34','gptq_W4g128_L63d8_asym15_Afp8','awq_gptq_W4g128_L63d8_asym15_Afp8','ours_W4A4KV4_g32','mix_asym_W3g64_late44_Afp8']
R={}
for d in sorted(glob.glob('/models/tmp/wsp2/results/p4/*/')):
    k=d.rstrip('/').split('/')[-1]; runs=[json.load(open(f)) for f in sorted(glob.glob(d+'s*.json'))]
    wt=[r['wikitext2']/BF['wt']*100-100 for r in runs]; fw=[r['fw25test']/BF['fw']*100-100 for r in runs]; zs=[r['zs_avg'] for r in runs]
    m=lambda x:sum(x)/len(x); sd=lambda x:(st.stdev(x) if len(x)>1 else 0.0)
    r0=runs[0]
    passm = m(wt)<=BAR and m(fw)<=BAR and m(zs)>=ZG
    worst = max(wt)<=BAR and max(fw)<=BAR
    R[k]=dict(n=len(runs),W=r0['wbits'],A=r0['abits_flopw'],B=r0['bitops'],KV=r0['kvbits'],wt=m(wt),fw=m(fw),zs=m(zs),sdwt=sd(wt),sdfw=sd(fw),sdzs=sd(zs),
              wtmax=max(wt),fwmax=max(fw),zsmin=min(zs),pass_mean=passm,pass_worst=worst,final=k in FIN)
for k,r in sorted(R.items(),key=lambda x:(not x[1]['final'],x[1]['B'])):
    print(f"{'F' if r['final'] else 'B'} {k:34s} n={r['n']} W={r['W']:.3f} A={r['A']:.2f} B={r['B']:.1f} KV={r['KV']:.2f} | wt {r['wt']:+.2f}±{r['sdwt']:.2f} fw {r['fw']:+.2f}±{r['sdfw']:.2f} zs {r['zs']:.2f}±{r['sdzs']:.2f} | max {r['wtmax']:+.2f} {r['fwmax']:+.2f} minzs {r['zsmin']:.2f} | mean:{'PASS' if r['pass_mean'] else 'fail'} worst:{'ok' if r['pass_worst'] else 'no'}")
PB=[k for k,r in R.items() if not r['final'] and r['pass_mean']]
print('passing baselines',PB)
mb=min(R[k]['W'] for k in PB); print('M base (min W of passing baselines)',mb, [k for k in PB if R[k]['W']==mb])
bB=[k for k in PB if R[k]['A']<16]; print('B bases', [(k,R[k]['W'],R[k]['A'],R[k]['B']) for k in bB])
for k in FIN:
    r=R[k]; ok=r['pass_mean'] and r['pass_worst']
    Mwin = ok and r['W']<=mb-0.5
    Bwin = ok and all((R[b]['W']-r['W']>=0.5 and r['A']<=R[b]['A']) or (R[b]['A']-r['A']>=0.5 and r['W']<=R[b]['W']) for b in bB) and r['A']<16
    best_bB=min(R[b]['B'] for b in bB)
    print(f"{k:34s} pass={ok} Mwin={Mwin} (W {r['W']:.3f} vs {mb:.3f}) Bwin={Bwin} (B {r['B']:.1f} vs {best_bB:.1f}, ratio {r['B']/best_bB:.2f}; WxA {r['W']*r['A']:.1f})")
json.dump(R,open('/models/tmp/wsp2/results/p4_summary/summary.json','w'),indent=1)
