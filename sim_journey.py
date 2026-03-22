import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(7)

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

TOPICS = [
    {"name": "Kinematics",          "W": 0.80, "D": 0.50, "prereqs": []},
    {"name": "Newton's Laws",       "W": 0.90, "D": 0.60, "prereqs": [0]},
    {"name": "Work Power Energy",   "W": 0.85, "D": 0.65, "prereqs": [1]},
    {"name": "Rotational Motion",   "W": 0.85, "D": 0.80, "prereqs": [2]},
    {"name": "Gravitation",         "W": 0.70, "D": 0.60, "prereqs": [1]},
    {"name": "Thermodynamics",      "W": 0.80, "D": 0.70, "prereqs": []},
    {"name": "Kinetic Theory",      "W": 0.70, "D": 0.65, "prereqs": [5]},
    {"name": "Waves",               "W": 0.75, "D": 0.60, "prereqs": []},
    {"name": "Optics",              "W": 0.85, "D": 0.75, "prereqs": [7]},
    {"name": "Electrostatics",      "W": 0.90, "D": 0.75, "prereqs": []},
    {"name": "Current Electricity", "W": 0.85, "D": 0.70, "prereqs": [9]},
    {"name": "Magnetism",           "W": 0.80, "D": 0.75, "prereqs": [10]},
    {"name": "EMI",                 "W": 0.75, "D": 0.80, "prereqs": [11]},
    {"name": "Modern Physics",      "W": 0.85, "D": 0.80, "prereqs": []},
    {"name": "Semiconductor",       "W": 0.70, "D": 0.65, "prereqs": []},
]
N     = len(TOPICS)
W_arr = np.array([t["W"] for t in TOPICS])
D_arr = np.array([t["D"] for t in TOPICS])
DAYS  = 180

def gen_compliance():
    seq = []
    for d in range(DAYS):
        if d in [20, 21, 22]: seq.append(0.0)
        elif d in [85, 86, 87, 88]: seq.append(0.0)
        else:
            r = np.random.random()
            if r < 0.55:   seq.append(1.0)
            elif r < 0.75: seq.append(float(np.random.uniform(0.2, 0.7)))
            else:          seq.append(0.0)
    return seq

COMP = gen_compliance()
M = np.full(N, 0.2); n = np.zeros(N, int); done = np.zeros(N, bool)
K0=2.0; Kt=7.0; rb=0.05; r=0.05; rp=0.05
K=K0; cm=0; sd=[]; resets=[]
lK=[]; lKth=[]; lD=[]; lH=[]

for day in range(DAYS):
    c = COMP[day]
    if cm >= 3: resets.append(day); cm=0; r=rb
    re=max(r,rb*0.3); Kth=min(K0*(1+rb)**day,Kt); lKth.append(Kth); st=[]
    if c > 0:
        h=c*min(K0*(1+re)**day,Kt); cm=0
        P=W_arr*D_arr*(1.0-M)
        for i in range(N):
            if M[i]>=0.82: done[i]=True
            if M[i]<0.62 and done[i]: done[i]=False
            if done[i]: P[i]=0.0
        P2=P.copy()
        for j in range(N):
            for p in TOPICS[j]["prereqs"]: P2[p]+=0.9*0.7*P[j]
        P=P2
        elig=[i for i in range(N) if all(M[p]>=0.28 for p in TOPICS[i]["prereqs"]) and not done[i]]
        if not elig: elig=[i for i in range(N) if not done[i]]
        if not elig: elig=list(range(N))
        es=sorted(elig,key=lambda i:P[i],reverse=True)
        nt=min(len(es),max(2,int(h/1.5))); top=es[:nt]
        pv=np.array([P[i]+1e-9 for i in top]); pv/=pv.sum(); he=pv*h
        for idx,t in enumerate(top):
            if he[idx]<0.18: continue
            if n[t]<=3: s=np.random.uniform(0.38,0.58)
            elif n[t]<=8: s=np.random.uniform(0.48,0.75)
            else: s=np.random.uniform(0.58,0.95)
            n[t]+=1; a=1.0/n[t]; M[t]=np.clip(M[t]+a*(s-M[t]),0,1); st.append(t)
        dl=[i for i in range(N) if done[i]]
        if day%5==0 and dl and h>0:
            for rev in sorted(dl,key=lambda i:M[i])[:2]:
                s=np.random.uniform(0.75,0.95); n[rev]+=1; a=1.0/n[rev]
                M[rev]=np.clip(M[rev]+a*(s-M[rev]),0,1); st.append(rev)
        D=np.clip(h/Kth,0,1) if Kth>0 else 1.0
        if D>=0.8: r=max(rp,rb)
        elif D>0: rp=r; r=max(r/2,rb*0.3)
        sd.append(D)
        if len(sd)>7: sd.pop(0)
        if len(sd)==7 and all(d>=0.8 for d in sd): r*=1.1
    else:
        h=0.0; D=0.0; cm+=1; rp=r; r=rb*0.1
        sd.append(0.0)
        if len(sd)>7: sd.pop(0)
    for j in range(N):
        if j not in st: M[j]*=0.995
    M=np.clip(M,0,1); K=0.8*K+0.2*h
    lK.append(float(K)); lH.append(float(h)); lD.append(float(D))

dx=np.arange(DAYS)
s7=lambda x: pd.Series(x).rolling(7,min_periods=1).mean().tolist()
print(f"Resets fired: {resets}")
print(f"Final capacity: {lK[-1]:.2f} hrs/day")

# Figure 1 — 8x3.5 for article twocolumn full width
fig1,ax1=plt.subplots(figsize=(8,3.5))
ax1.fill_between(dx,s7(lH),alpha=0.18,color='#16A34A')
ax1.plot(dx,s7(lH),color='#16A34A',lw=2.2,label='Actual hours studied (7-day avg)')
pd1=next((i for i,k in enumerate(lKth) if k>=6.99),DAYS)
ax1.plot(dx[:pd1],lKth[:pd1],color='black',lw=1.5,ls='--',alpha=0.5,label='K(t) theoretical')
ax1.axhline(K0,color='#6B7280',lw=1.2,ls=':',alpha=0.6,label=f'K\u2080 = {K0} hrs (baseline)')
for rd in resets:
    ax1.axvline(rd,color='#DC2626',lw=1.4,ls=':',alpha=0.7)
    ax1.annotate('Reset',xy=(rd+1.5,0.35),fontsize=10,color='#DC2626',fontstyle='italic')
ax1.set_xlabel("Day"); ax1.set_ylabel("Hours per Day")
ax1.set_xlim(0,DAYS); ax1.set_ylim(0,8.5)
ax1.legend(loc='upper left',fontsize=11,framealpha=0.9)
fig1.tight_layout(pad=0.6)
fig1.savefig("capacity_recovery.png",dpi=300,bbox_inches='tight')
plt.close(fig1); print("Saved capacity_recovery.png")

# Figure 2
fig2,ax2=plt.subplots(figsize=(8,3.2))
ax2.fill_between(dx,s7(lD),alpha=0.15,color='#2563EB')
ax2.plot(dx,s7(lD),color='#2563EB',lw=2.2,label='D = Actual / Scheduled hours (7-day avg)')
for rd in resets:
    ax2.axvline(rd,color='#DC2626',lw=1.4,ls=':',alpha=0.7)
    ax2.annotate('Reset',xy=(rd+1.5,0.06),fontsize=10,color='#DC2626',fontstyle='italic')
ax2.set_xlabel("Day"); ax2.set_ylabel("Discipline Score (0\u20131)")
ax2.set_xlim(0,DAYS); ax2.set_ylim(0,1.1)
ax2.legend(loc='lower right',fontsize=11,framealpha=0.9)
fig2.tight_layout(pad=0.6)
fig2.savefig("discipline_journey.png",dpi=300,bbox_inches='tight')
plt.close(fig2); print("Saved discipline_journey.png")
print("\nAll journey figures saved.")
