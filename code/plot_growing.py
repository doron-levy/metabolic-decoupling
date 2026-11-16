import os
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
hs=np.load("held_sweep.npz"); sn=np.load("snapshots.npz"); td=np.load("tdec.npz")
z=sn["z"]

fig,ax=plt.subplots(2,2,figsize=(11,8.4)); fig.subplots_adjust(hspace=0.33,wspace=0.27)

# (a) held-supply separation trajectories
a=ax[0,0]
cols={'0.02':'#1a5276','0.04':'#2874a6','0.06':'#5499c7','0.08':'#e67e22','0.12':'#c0392b'}
for al in ['0.02','0.04','0.06','0.08','0.12']:
    t=hs['t'+al]; s=hs['s'+al]
    lab=f'$\\alpha={al}$'+(' (coord.)' if float(al)<0.07 else ' (decoupled)')
    a.plot(t,s,lw=2,color=cols[al],label=lab)
a.axhline(2.5,ls=':',color='gray',lw=1); a.text(1,2.6,'decoupling threshold',fontsize=7.5,color='gray')
a.set_xlabel(r'time $t/\tau$'); a.set_ylabel(r'separation $\Delta(t)/\xi_\ell$')
a.set_title('(a) held supply: separation settles below a threshold, runs away above',fontsize=10)
a.legend(fontsize=7.5,loc='upper center',ncol=2,framealpha=0.9); a.set_ylim(0,3.5)

# (b) coordinated snapshot (each field normalized to its own peak for display)
b=ax[0,1]
nrm=lambda v: v/max(v.max(),1e-9)
b.plot(z,nrm(sn['uC']),color='#16a085',lw=1.8,label='glycolytic $u$ (norm.)')
b.plot(z,nrm(sn['wC']),color='#8e44ad',lw=1.8,label='oxidative $w$ (norm.)')
b.plot(z,nrm(sn['lC']),color='#27ae60',lw=1.3,label='lactate (norm.)')
b.plot(z,np.clip(0.5*(1+sn['mC']),0,1),color='#e67e22',lw=1.3,ls='--',label='oxidative activity')
b.set_xlim(-3,3); b.set_ylim(-0.05,1.62); b.set_xlabel('comoving position'); b.set_ylabel('field (normalized)')
b.set_title('(b) coordinated: comigrating symbiotic pair',fontsize=10); b.legend(fontsize=7.5,loc='upper right',framealpha=0.9)

# (c) decoupled snapshot (each field normalized to its own peak for display)
c=ax[1,0]
c.plot(z,nrm(sn['uD']),color='#16a085',lw=1.8,label='glycolytic $u$ (norm.)')
c.plot(z,nrm(sn['wD']),color='#8e44ad',lw=1.8,label='oxidative $w$ (now glycolytic)')
c.plot(z,nrm(sn['lD']),color='#27ae60',lw=1.3,label='lactate (norm.)')
c.plot(z,np.clip(0.5*(1+sn['mD']),0,1),color='#e67e22',lw=1.3,ls='--',label='oxidative activity')
c.set_xlim(-3,3); c.set_ylim(-0.05,1.62); c.set_xlabel('comoving position'); c.set_ylabel('field (normalized)')
c.set_title('(c) decoupled: pair pulled apart, consumer starved',fontsize=10); c.legend(fontsize=7.5,loc='upper right',framealpha=0.9)

# (d) decoupling time vs growth rate: held (sharp threshold) vs dilution (soft)
d=ax[1,1]
hk=np.array(td['hk']);hv=np.array(td['hv']); dk=np.array(td['dk']);dv=np.array(td['dv'])
oh=np.argsort(hk); od=np.argsort(dk)
d.plot(hk[oh],hv[oh],'o-',color='#c0392b',ms=5,lw=1.6,label='supply held (sharp threshold)')
d.plot(dk[od],dv[od],'s-',color='#27ae60',ms=5,lw=1.6,label='dilution (soft)')
d.axvspan(0.0,0.07,color='#c0392b',alpha=0.06)
d.axvline(0.07,color='#c0392b',ls=':',lw=1.2); d.text(0.072,45,r'held $\alpha^\ast$',color='#c0392b',fontsize=9)
d.annotate('held: coordinated\n(no decoupling)',xy=(0.035,30),fontsize=7.5,color='#a03020',ha='center')
d.set_xlabel(r'growth rate $\alpha\tau$'); d.set_ylabel(r'decoupling time $t_{\rm dec}/\tau$')
d.set_title('(d) full PDE: held supply has a threshold, dilution does not',fontsize=10)
d.legend(fontsize=8.5); d.set_xlim(0,0.21); d.set_ylim(0,58)

os.makedirs('../figures',exist_ok=True)
fig.savefig("../figures/shuttle_growing.pdf",bbox_inches="tight")
print("saved shuttle_growing.pdf")
