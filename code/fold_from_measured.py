import os
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import full_solver as F

d=np.load("Gmeasured.npz"); Db,Gb,Dm,Gm=d["Db"],d["Gb"],d["Dm"],d["Gm"]
def Gfun(D): return np.interp(D,Db,Gb,left=Gb[0],right=0.0)
astar=np.max(Gb[Db>0.7]/Db[Db>0.7]); astar_m=np.max(Gm[Dm>0.7]/Dm[Dm>0.7])
print("alpha* bistable=%.4f  monostable=%.4f (near zero -> no coordinated state)"%(astar,astar_m))

def tdec(alpha,dilute=False,D0=0.8,Desc=5.5,dt=1e-3,Tmax=20000):
    D=D0;t=0.0
    while t<Tmax:
        clos=Gfun(D)*(np.exp(-alpha*t) if dilute else 1.0)
        D+=dt*(alpha*D-clos); t+=dt
        if D>Desc: return t
        if D<1e-3: D=1e-3
    return np.nan

eps=np.array([0.0015,0.0025,0.004,0.006,0.009,0.014,0.021,0.032,0.05,0.075])
th=np.array([tdec(astar+e) for e in eps]); sh=np.polyfit(np.log(eps),np.log(th),1)[0]
als=np.array([0.005,0.008,0.011,0.015,0.020,0.026,0.033,0.042,0.052,0.059])
td=np.array([tdec(a,dilute=True) for a in als])
sc=np.polyfit(np.log(astar/als),als*td,1)[0]
print("HELD exponent=%.3f (predict -0.5)   DILUTE collapse slope=%.3f (predict ~1)"%(sh,sc))

# field snapshot at a coordinated separation
x,dx=F.grid(F.N); Dsnap=1.2
u=F.gaussian(x,F.Xc-Dsnap/2); w=F.gaussian(x,F.Xc+Dsnap/2)
m=np.ones(F.N)
for _ in range(80):
    l=F.lactate(u,w,m,x,dx); mn=F.m_branch(l,m)
    if np.max(np.abs(mn-m))<1e-9: m=mn;break
    m=mn
l=F.lactate(u,w,m,x,dx)

fig,ax=plt.subplots(2,2,figsize=(11,8.2)); fig.subplots_adjust(hspace=0.32,wspace=0.27)

a=ax[0,0]
a.axhline(0,color='k',lw=0.6,alpha=0.5)
a.plot(Db,Gb,color='#c0392b',lw=2.2,label='bistable phenotype')
a.plot(Dm,Gm,color='#2980b9',lw=1.8,ls='--',label='monostable phenotype')
a.plot(Db,astar*Db,color='gray',lw=1.1,ls=':',label=r'$\alpha^\ast\Delta$ (tangent)')
a.set_xlim(0,5); a.set_xlabel(r'separation $\Delta/\xi_\ell$'); a.set_ylabel(r'measured closing rate $G(\Delta)$')
a.set_title('(a) $G$ measured from the full operator: hump needs bistability',fontsize=10.3)
a.legend(fontsize=8.5,loc='upper left')

b=ax[0,1]
b.plot(x-F.Xc,u,color='#16a085',lw=1.8,label='glycolytic source $u$')
b.plot(x-F.Xc,w,color='#8e44ad',lw=1.8,label='oxidative consumer $w$')
b.plot(x-F.Xc,l/l.max(),color='#27ae60',lw=1.4,ls='-',label='lactate (norm.)')
b.plot(x-F.Xc,F.fo(m),color='#e67e22',lw=1.4,ls='--',label='oxidative activity $f_o(m)$')
b.set_xlim(-4,4); b.set_ylim(-0.05,1.5); b.set_xlabel(r'$x-x_c$'); b.set_ylabel('field'); 
b.set_title(r'(b) two-cluster setup at coordinated $\Delta$',fontsize=10.3)
b.legend(fontsize=7.5,loc='upper left')

c=ax[1,0]
c.loglog(eps,th,'o',color='#c0392b',ms=6)
xx=np.logspace(np.log10(eps.min()),np.log10(eps.max()),40)
c.loglog(xx,th[0]*(xx/eps[0])**(-0.5),'k--',lw=1.2,label='slope $-1/2$')
c.set_xlabel(r'$\alpha-\alpha^\ast$'); c.set_ylabel(r'$t_{\rm dec}/\tau$')
c.set_title(f'(c) held supply, measured $G$: critical slowing {sh:.2f}',fontsize=10.3); c.legend(fontsize=8.5)

dd=ax[1,1]
dd.plot(np.log(astar/als),als*td,'s',color='#27ae60',ms=6)
xl=np.linspace(np.log(astar/als).min(),np.log(astar/als).max(),20)
dd.plot(xl,np.polyval(np.polyfit(np.log(astar/als),als*td,1),xl),'k--',lw=1.2,label=f'slope {sc:.2f}')
dd.set_xlabel(r'$\ln(\alpha^\ast/\alpha)$'); dd.set_ylabel(r'$\alpha\,t_{\rm dec}/\tau$')
dd.set_title('(d) dilution, measured $G$: $1/\\alpha$ slow passage',fontsize=10.3); dd.legend(fontsize=8.5)

os.makedirs('../figures',exist_ok=True)
fig.savefig("../figures/shuttle_operator.pdf",bbox_inches="tight")
np.savez("operator_fold.npz",astar=astar,astar_m=astar_m,eps=eps,th=th,als=als,td=td,sh=sh,sc=sc)
print("saved shuttle_operator.pdf")
