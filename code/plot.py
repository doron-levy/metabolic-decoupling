import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shuttle as S

sw=np.load("sweeps.npz"); sn=np.load("snaps.npz")
Ld,Jd,Lh,Jh=sw["Ld"],sw["Jd"],sw["Lh"],sw["Jh"]
xh=np.linspace(0,1,len(sn["phiS"]))

# dynamic runs (recompute clean)
alphas=[0.03,0.08,0.15,0.30]; dyn={}
for al in alphas:
    ts,Ls,Js,Ms=S.grow(al,Tmax=np.log(8.0/S.L0)/al); dyn[al]=(ts,Ls,Js)

fig,ax=plt.subplots(2,2,figsize=(11,8.2)); fig.subplots_adjust(hspace=0.32,wspace=0.28)

# (a) static severing curve: dilution vs supply held
a=ax[0,0]
a.plot(Ld,Jd,'o-',color='#c0392b',ms=4,lw=1.8,label='growth dilutes lactate supply')
a.plot(Lh,Jh,'s-',color='#2471a3',ms=4,lw=1.8,label='supply held (proliferation compensates)')
a.axhline(0.5*Jd[0],ls=':',color='gray',lw=1); a.axvline(S.L0,ls='--',color='k',lw=0.8,alpha=0.5)
a.set_xlabel('tumor size  $L\\,/\\,\\xi_\\ell$'); a.set_ylabel('lactate shuttle flux  $J$')
a.set_title('(a) growth severs the shuttle, but only when it dilutes supply',fontsize=10.5)
a.legend(fontsize=8.5,loc='upper right'); a.set_ylim(0,0.26)
a.text(1.55,0.505*Jd[0]+0.004,'half-max',fontsize=7.5,color='gray')

# (b) fields when coordinated (small L) — oxidative shell present
b=ax[0,1]
b.plot(xh,sn["phiS"],color='#8e44ad',lw=2,label='phenotype $s$ ($+1$ glycolytic)')
b.plot(xh,sn["lS"]/sn["lS"].max(),color='#27ae60',lw=1.6,ls='-',label='lactate (norm.)')
b.plot(xh,sn["oS"],color='#e67e22',lw=1.6,ls='--',label='oxygen')
b.axhline(0,color='k',lw=0.6,alpha=0.4)
b.fill_between(xh,-1,sn["phiS"],where=sn["phiS"]>0,color='#8e44ad',alpha=0.12)
b.set_title('(b) small tumor ($L=1.5\\,\\xi_\\ell$): oxidative shell present',fontsize=10.5)
b.set_xlabel('reference position  $\\hat x$   (core $0$ $\\to$ rim $1$)'); b.set_ylabel('field')
b.legend(fontsize=8,loc='lower left'); b.set_ylim(-1.5,1.2)

# (c) fields when severed (large L) — shell gone
c=ax[1,0]
c.plot(xh,sn["phiL"],color='#8e44ad',lw=2,label='phenotype $s$')
c.plot(xh,sn["lL"]/max(sn["lS"].max(),1e-9),color='#27ae60',lw=1.6,label='lactate (same norm.)')
c.plot(xh,sn["oL"],color='#e67e22',lw=1.6,ls='--',label='oxygen')
c.axhline(0,color='k',lw=0.6,alpha=0.4)
c.fill_between(xh,-1,sn["phiL"],where=sn["phiL"]>0,color='#8e44ad',alpha=0.12)
c.set_title('(c) large tumor ($L=6\\,\\xi_\\ell$): shuttle severed, shell gone',fontsize=10.5)
c.set_xlabel('reference position  $\\hat x$   (core $0$ $\\to$ rim $1$)'); c.set_ylabel('field')
c.legend(fontsize=8,loc='upper left'); c.set_ylim(-1.1,1.15)

# (d) dynamic: shuttle collapse in time, faster for faster growth
d=ax[1,1]
cols=plt.cm.viridis(np.linspace(0.15,0.85,len(alphas)))
for al,col in zip(alphas,cols):
    ts,Ls,Js=dyn[al]; d.plot(ts,Js,lw=2,color=col,label=f'$\\alpha={al}$')
d.set_xlabel('time  $t\\,/\\,\\tau$'); d.set_ylabel('lactate shuttle flux  $J(t)$')
d.set_title('(d) a growing tumor severs it in real time (faster if it grows faster)',fontsize=10.5)
d.legend(fontsize=8.5,title='growth rate'); d.set_xlim(0,60); d.set_ylim(0,0.18)

os.makedirs('../figures',exist_ok=True)
fig.savefig("../figures/shuttle_severing.pdf",bbox_inches="tight")
print("saved figure")
# report numbers
MoxS=np.trapezoid(np.clip(0.5*(1-sn["phiS"]),0,1),dx=xh[1]-xh[0])
MoxL=np.trapezoid(np.clip(0.5*(1-sn["phiL"]),0,1),dx=xh[1]-xh[0])
print(f"oxidative shell mass: L=1.5 -> {MoxS:.3f} ;  L=6 -> {MoxL:.3f}")
print(f"static shuttle: dilution J(1->8) {Jd[0]:.3f}->{Jd[-1]:.3f} (x{Jd[-1]/Jd[0]:.2f}); held {Jh[0]:.3f}->{Jh[-1]:.3f} (x{Jh[-1]/Jh[0]:.2f})")
