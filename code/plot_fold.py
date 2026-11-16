import os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import reduced_fold as rf

D_b,G_b,m_b=rf.D_b,rf.G_b,rf.m_b
D_m,G_m,m_m=rf.D_m,rf.G_m,rf.m_m
astar=rf.astar_b; astar_m=rf.astar_m

# refined held-supply critical slowing (points close to alpha*)
eps=np.array([0.0015,0.003,0.006,0.012,0.025,0.05,0.1,0.2])
th=np.array([rf.t_decouple(astar+e,D_b,G_b,dilute=False) for e in eps])
close=eps<=0.03
slope_hold=np.polyfit(np.log(eps[close]),np.log(th[close]),1)[0]

# held t_dec vs alpha (for the headline contrast) ; nan below alpha* (coordinated forever)
al_hold=np.concatenate([np.linspace(0.4,astar-0.04,6),
                        astar+np.array([0.006,0.015,0.03,0.06,0.12,0.22,0.38,0.6,0.85,1.15])])
t_hold=np.array([rf.t_decouple(a,D_b,G_b,dilute=False) for a in al_hold])

# dilution slow passage
als=np.array([0.04,0.06,0.09,0.13,0.2,0.3,0.45,0.65])
td=np.array([rf.t_decouple(a,D_b,G_b,dilute=True) for a in als])
# slope-one collapse: alpha*t_dec vs ln(alpha*/alpha)
x_col=np.log(astar/als); y_col=als*td
slope_col=np.polyfit(x_col,y_col,1)[0]

fig,ax=plt.subplots(2,2,figsize=(11,8.4)); fig.subplots_adjust(hspace=0.33,wspace=0.27)

# (a) closing rate: bistable cliff vs monostable smooth
a=ax[0,0]
a.axhline(0,color='k',lw=0.6,alpha=0.5)
a.plot(D_b,G_b,color='#c0392b',lw=2.2,label='bistable phenotype (sharp cliff)')
a.plot(D_m,G_m,color='#2980b9',lw=1.8,ls='--',label='monostable (smooth)')
a.plot(D_b,astar*D_b,color='gray',lw=1.2,ls=':',label=r'growth line $\alpha^\ast\,\Delta$ (tangent)')
a.set_xlim(0,3); a.set_ylim(-0.6,0.7)
a.set_xlabel(r'core$-$shell separation  $\Delta/\xi_\ell$'); a.set_ylabel(r'metabolic closing rate  $G(\Delta)$')
a.set_title('(a) bistable phenotype carves a hump with a cliff',fontsize=10.5)
a.legend(fontsize=8,loc='lower right')

# (b) headline: decoupling time vs growth rate -- held (sharp, finite alpha*) vs dilution (soft)
b=ax[0,1]
b.plot(al_hold,t_hold,'o-',color='#c0392b',ms=3,lw=1.6,label='supply held: sharp fold at $\\alpha^\\ast$')
b.plot(als,td,'s-',color='#27ae60',ms=4,lw=1.6,label='dilution: soft, any $\\alpha>0$ decouples')
b.axvline(astar,color='#c0392b',ls=':',lw=1.2); b.text(astar+0.03,120,r'$\alpha^\ast$',color='#c0392b',fontsize=11)
b.set_xlabel(r'growth rate  $\alpha\tau$'); b.set_ylabel(r'decoupling time  $t_{\rm dec}/\tau$')
b.set_ylim(0,160); b.set_xlim(0,2.2)
b.set_title('(b) growth severs symbiosis: sharp threshold vs soft crossover',fontsize=10.5)
b.legend(fontsize=8.5,loc='upper right')

# (c) held critical slowing exponent
c=ax[1,0]
c.loglog(eps,th,'o',color='#c0392b',ms=6)
xx=np.logspace(np.log10(eps.min()),np.log10(eps.max()),50)
c.loglog(xx, th[0]*(xx/eps[0])**(-0.5),'k--',lw=1.2,label='slope $-1/2$ (saddle-node)')
c.set_xlabel(r'$\alpha-\alpha^\ast$'); c.set_ylabel(r'$t_{\rm dec}/\tau$')
c.set_title(f'(c) held supply: critical slowing, fitted {slope_hold:.2f}',fontsize=10.5)
c.legend(fontsize=8.5)

# (d) dilution slope-one collapse
d=ax[1,1]
d.plot(x_col,y_col,'s',color='#27ae60',ms=6)
xl=np.linspace(x_col.min(),x_col.max(),20)
d.plot(xl, np.polyval(np.polyfit(x_col,y_col,1),xl),'k--',lw=1.2,label=f'slope {slope_col:.2f} ($\\approx 1$)')
d.set_xlabel(r'$\ln(\alpha^\ast/\alpha)$'); d.set_ylabel(r'$\alpha\,t_{\rm dec}/\tau$')
d.set_title('(d) dilution: leading $1/\\alpha$ slow-passage collapse',fontsize=10.5)
d.legend(fontsize=8.5)

os.makedirs('../figures',exist_ok=True)
fig.savefig("../figures/shuttle_fold.pdf",bbox_inches="tight")
print("alpha* = %.3f"%astar)
print("held critical-slowing exponent (eps<=0.03 fit) = %.3f  (predict -0.5)"%slope_hold)
print("dilution collapse slope = %.3f  (predict ~1)"%slope_col)
print("saved shuttle_fold.pdf")
