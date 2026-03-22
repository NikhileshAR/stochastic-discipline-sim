import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.18, 'grid.linestyle': '--',
    'axes.labelsize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'legend.framealpha': 0.92,
    'figure.facecolor': '#FFFFFF', 'axes.facecolor': '#F9F9F9',
})

COL={'A':'#DC2626','B':'#2563EB','C':'#16A34A'}
TOPICS=[
    {"name":"Kinematics","W":0.80,"D":0.50,"prereqs":[]},
    {"name":"Newton's Laws","W":0.90,"D":0.60,"prereqs":[0]},
    {"name":"Work Power Energy","W":0.85,"D":0.65,"prereqs":[1]},
    {"name":"Rotational Motion","W":0.85,"D":0.80,"prereqs":[2]},
    {"name":"Gravitation","W":0.70,"D":0.60,"prereqs":[1]},
    {"name":"Properties of Matter","W":0.60,"D":0.55,"prereqs":[]},
    {"name":"Thermodynamics","W":0.80,"D":0.70,"prereqs":[]},
    {"name":"Kinetic Theory","W":0.70,"D":0.65,"prereqs":[6]},
    {"name":"Waves","W":0.75,"D":0.60,"prereqs":[]},
    {"name":"Optics","W":0.85,"D":0.75,"prereqs":[8]},
    {"name":"Electrostatics","W":0.90,"D":0.75,"prereqs":[]},
    {"name":"Current Electricity","W":0.85,"D":0.70,"prereqs":[10]},
    {"name":"Magnetism","W":0.80,"D":0.75,"prereqs":[11]},
    {"name":"EMI","W":0.75,"D":0.80,"prereqs":[12]},
    {"name":"AC Circuits","W":0.70,"D":0.75,"prereqs":[13]},
    {"name":"EM Waves","W":0.60,"D":0.50,"prereqs":[]},
    {"name":"Ray Optics","W":0.80,"D":0.70,"prereqs":[9]},
    {"name":"Wave Optics","W":0.75,"D":0.75,"prereqs":[8]},
    {"name":"Modern Physics","W":0.85,"D":0.80,"prereqs":[15]},
    {"name":"Semiconductor","W":0.70,"D":0.65,"prereqs":[]},
]
N=len(TOPICS); W_arr=np.array([t["W"] for t in TOPICS]); D_arr=np.array([t["D"] for t in TOPICS])
DAYS=180; HIGH_W=W_arr>=0.75

def gen_compliance():
    seq=[]
    for _ in range(DAYS):
        r=np.random.random()
        if r<0.70: seq.append(1.0)
        elif r<0.85: seq.append(float(np.random.uniform(0.3,0.8)))
        else: seq.append(0.0)
    return seq

COMP=gen_compliance()
COMP[25]=0.0;COMP[26]=0.0;COMP[27]=0.0
COMP[105]=0.0;COMP[106]=0.0;COMP[107]=0.0

def wm(M): return float(np.sum(W_arr[HIGH_W]*M[HIGH_W])/np.sum(W_arr[HIGH_W]))

def run_A():
    M=np.full(N,0.2);n=np.zeros(N,int);mlog=[];hlog=[]
    for day in range(DAYS):
        comp=COMP[day]
        if comp>0:
            h=comp*2.0; weights=np.random.dirichlet(np.ones(N)*0.3)
            t=np.random.choice(N,p=weights); s=np.random.uniform(0.38,0.62)
            n[t]+=1; a=1.0/n[t]; M[t]=np.clip(M[t]+a*(s-M[t]),0,1)
            for j in range(N):
                if j!=t: M[j]*=0.997
        else:
            h=0.0; M[:]=np.clip(0.997*M,0,1)
        hlog.append(h);mlog.append(wm(M))
    return {"M":M,"m":mlog,"tot":sum(hlog)}

def run_B():
    M=np.full(N,0.2);n=np.zeros(N,int);ptr=0;mlog=[];hlog=[]
    for day in range(DAYS):
        c=COMP[day]
        if c>0:
            h=c*4.0;t=ptr%N;s=np.random.uniform(0.45,0.78)
            n[t]+=1;a=1.0/n[t];M[t]=np.clip(M[t]+a*(s-M[t]),0,1)
            if n[t]>=5: ptr+=1
            for j in range(N):
                if j!=t: M[j]*=0.995
        else:
            h=0.0;M[:]=np.clip(0.995*M,0,1)
        hlog.append(h);mlog.append(wm(M))
    return {"M":M,"m":mlog,"tot":sum(hlog)}

def run_C():
    M=np.full(N,0.2);n=np.zeros(N,int);done=np.zeros(N,bool)
    K0=2.0;Kt=7.0;rb=0.05;r=0.05;rp=0.05;K=K0;cm=0;sd=[];Hc=0.0;Hp=0.0
    resets=[];mlog=[];hlog=[]
    for day in range(DAYS):
        c=COMP[day]
        if cm>=3: resets.append(day);cm=0;r=rb
        Kth=min(K0*(1+rb)**day,Kt);st=[]
        if c>0:
            h=c*Kth;cm=0;P=W_arr*D_arr*(1.0-M)
            for i in range(N):
                if M[i]>=0.65: done[i]=True
                if M[i]<0.48 and done[i]: done[i]=False
                if done[i]: P[i]=0.0
            P2=P.copy()
            for j in range(N):
                for p in TOPICS[j]["prereqs"]: P2[p]+=0.9*0.7*P[j]
            P=P2
            elig=[i for i in range(N) if all(M[p]>=0.15 for p in TOPICS[i]["prereqs"]) and not done[i]]
            if not elig: elig=[i for i in range(N) if not done[i]]
            if not elig: elig=list(range(N))
            neglected=[i for i in range(N) if M[i]<=0.20+1e-6]
            if neglected and h>2.5: elig=list(set(elig+neglected[:3]))
            es=sorted(elig,key=lambda i:P[i],reverse=True)
            nt2=min(len(es),max(2,int(h/1.5)));top=es[:nt2]
            pv=np.array([P[i]+1e-9 for i in top]);pv/=pv.sum();he=pv*h
            for idx,t in enumerate(top):
                if he[idx]<0.18: continue
                s=np.random.uniform(0.45,0.75);n[t]+=1;a=1.0/n[t]
                M[t]=np.clip(M[t]+a*(s-M[t]),0,1);st.append(t)
                if M[t]<0.5: Hc+=he[idx]
                else: Hp+=he[idx]
            dl=[i for i in range(N) if done[i]]
            if day%5==0 and dl and h>0:
                for rev in sorted(dl,key=lambda i:M[i])[:2]:
                    s=np.random.uniform(0.75,0.95);n[rev]+=1;a=1.0/n[rev]
                    M[rev]=np.clip(M[rev]+a*(s-M[rev]),0,1);st.append(rev)
            D=np.clip(h/Kth,0,1)
            if D>=0.8: r=max(rp,rb)
            elif D>0: rp=r;r=max(r/2,rb*0.3)
            sd.append(D)
            if len(sd)>7: sd.pop(0)
            if len(sd)==7 and all(d>=0.8 for d in sd): r*=1.1
        else:
            h=0.0;cm+=1;rp=r;r=rb*0.1;sd.append(0.0)
            if len(sd)>7: sd.pop(0)
        for j in range(N):
            if j not in st: M[j]*=0.995
        M=np.clip(M,0,1);K=0.8*K+0.2*h
        hlog.append(h);mlog.append(wm(M))
    return {"M":M,"m":mlog,"resets":resets,"tot":sum(hlog)}

print("Running benchmark...")
rA=run_A();rB=run_B();rC=run_C()
dx=np.arange(DAYS)
HIGH_W_IDX=[i for i in range(N) if W_arr[i]>=0.75];NH=len(HIGH_W_IDX)
cA=sum(1 for i in HIGH_W_IDX if rA["M"][i]>0.4)/NH*100
cB=sum(1 for i in HIGH_W_IDX if rB["M"][i]>0.4)/NH*100
cC=sum(1 for i in HIGH_W_IDX if rC["M"][i]>0.4)/NH*100
print(f"Coverage: A={cA:.1f}% B={cB:.1f}% C={cC:.1f}%")
print(f"Hours: A={rA['tot']:.0f} B={rB['tot']:.0f} C={rC['tot']:.0f}")
print(f"Mastery: A={rA['m'][-1]:.3f} B={rB['m'][-1]:.3f} C={rC['m'][-1]:.3f}")

s14=lambda x: pd.Series(x).rolling(14,min_periods=1).mean().tolist()

# Fig 3 — weighted mastery, full text width 8x3.5
fig3,ax3=plt.subplots(figsize=(8,3.5))
ax3.plot(dx,s14(rA["m"]),color=COL['A'],ls='--',lw=2.0,alpha=0.9,label='Condition A — No Schedule')
ax3.plot(dx,s14(rB["m"]),color=COL['B'],ls=':',lw=2.0,alpha=0.9,label='Condition B — Static Schedule (4 hrs/day)')
ax3.plot(dx,s14(rC["m"]),color=COL['C'],ls='-',lw=2.4,alpha=0.95,label='Condition C — Adaptive System (Proposed)')
for rd in rC["resets"]:
    ax3.axvline(rd,color=COL['C'],ls=':',lw=0.9,alpha=0.45)
    ax3.annotate('\u21ba',xy=(rd+1,0.03),fontsize=10,color=COL['C'],alpha=0.8)
ax3.set_xlabel("Day");ax3.set_ylabel("Weighted Mastery Score (0\u20131)")
ax3.set_xlim(0,DAYS);ax3.set_ylim(0,1.0)
ax3.legend(loc='lower right',fontsize=11,framealpha=0.9)
fig3.tight_layout(pad=0.6)
fig3.savefig("weighted_mastery.png",dpi=300,bbox_inches='tight')
plt.close(fig3);print("Saved weighted_mastery.png")

# Fig 4 — coverage bar, half width 5x4
fig4,ax4=plt.subplots(figsize=(5,4))
bars=ax4.bar(
    ['Cond. A\n(No Schedule)','Cond. B\n(Static, 4 hrs)','Cond. C\n(Adaptive)'],
    [cA,cB,cC],color=[COL['A'],COL['B'],COL['C']],
    edgecolor='white',linewidth=1.5,width=0.52,alpha=0.92)
for bar,val in zip(bars,[cA,cB,cC]):
    ax4.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,
             f'{val:.0f}%',ha='center',va='bottom',fontsize=13,fontweight='bold',color='#1F2937')
ax4.set_ylabel("High-Priority Coverage (%)");ax4.set_ylim(0,115)
fig4.tight_layout(pad=0.6)
fig4.savefig("coverage_bar.png",dpi=300,bbox_inches='tight')
plt.close(fig4);print("Saved coverage_bar.png")

# Fig 5 — per topic, full text width 14x4
si=np.argsort(W_arr)[::-1];labs=[TOPICS[i]["name"] for i in si]
fig5,ax5=plt.subplots(figsize=(14,4))
x=np.arange(N);w=0.28
ax5.bar(x-w,rA["M"][si],w,color=COL['A'],alpha=0.85,edgecolor='white',lw=0.8,label='Condition A — No Schedule')
ax5.bar(x,rB["M"][si],w,color=COL['B'],alpha=0.85,edgecolor='white',lw=0.8,label='Condition B — Static Schedule')
ax5.bar(x+w,rC["M"][si],w,color=COL['C'],alpha=0.90,edgecolor='white',lw=0.8,label='Condition C — Adaptive (Proposed)')
ax5.set_xlabel("Topic (sorted by weightage, highest to lowest)")
ax5.set_ylabel("Mastery Score (0\u20131)")
ax5.set_xticks(x);ax5.set_xticklabels(labs,rotation=40,ha='right',fontsize=9)
ax5.set_ylim(0,1.05);ax5.legend(loc='upper right',fontsize=11)
fig5.tight_layout(pad=0.6)
fig5.savefig("pertopic_mastery.png",dpi=300,bbox_inches='tight')
plt.close(fig5);print("Saved pertopic_mastery.png")
print("\nAll benchmark figures saved.")
