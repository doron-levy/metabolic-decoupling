import numpy as np, shuttle as S

# --- static sweeps: dilution ON (rho ~ 1/L) vs supply HELD (rho const) ---
def static_sweep(hold=False, N=400):
    Ls=np.linspace(1.0,8.0,29); J=[]
    for L in Ls:
        # temporarily override dilution by monkeypatching rho inside relax
        dx=1.0/(N-1); xh=np.linspace(0,1,N); o=S.oxygen(xh,L)
        rho = S.rho0 if hold else S.rho0*S.L0/L
        phi=-0.9*np.ones(N); dt=5e-3
        for k in range(40000):
            l=S.solve_lactate(phi,o,L,rho,N,dx); ps=S.phi_target(l,o); pn=S.step_phi(phi,ps,L,dt,dx)
            if np.max(np.abs(pn-phi))/dt<1e-9: phi=pn;break
            phi=pn
        l=S.solve_lactate(phi,o,L,rho,N,dx); _,Jv=S.observ(phi,l,o,L,dx); J.append(Jv)
    return Ls,np.array(J)

Ld,Jd=static_sweep(hold=False)
Lh,Jh=static_sweep(hold=True)
np.savez("sweeps.npz",Ld=Ld,Jd=Jd,Lh=Lh,Jh=Jh)
print("dilution ON : J(L=1)=%.3f  J(L=8)=%.3f  ratio=%.2f"%(Jd[0],Jd[-1],Jd[-1]/Jd[0]))
print("supply HELD : J(L=1)=%.3f  J(L=8)=%.3f  ratio=%.2f"%(Jh[0],Jh[-1],Jh[-1]/Jh[0]))

# --- dynamic growth: shuttle flux over time for several growth rates ---
runs={}
for al in [0.03,0.08,0.15,0.30]:
    ts,Ls,Js,Ms=S.grow(al,Tmax=np.log(8.0/S.L0)/al)
    runs[al]=(ts,Ls,Js,Ms)
    # decoupling "time": when J falls to 25% of its initial coordinated value
    J0=Js[0]; below=np.where(Js<0.25*J0)[0]
    tdec = ts[below[0]] if len(below) else np.nan
    print("alpha=%.2f  J0=%.3f  min J=%.3f  t(J<25%%)=%s"%(al,J0,Js.min(), f"{tdec:.1f}" if not np.isnan(tdec) else "not reached"))
np.savez("dynamic.npz",**{f"a{k}":np.array(v,dtype=object) for k,v in runs.items()})

# --- snapshots (fields) at small vs large L, dilution on ---
phiS,lS,oS,_=S.relax_static(1.5); phiL,lL,oL,_=S.relax_static(6.0)
np.savez("snaps.npz",phiS=phiS,lS=lS,oS=oS,phiL=phiL,lL=lL,oL=oL)

# --- grid convergence on one dynamic run ---
import numpy as np
def endJ(al,N):
    ts,Ls,Js,Ms=S.grow(al,Tmax=25.0,N=N); return Ls,Js
L2,J2=endJ(0.08,300); L4,J4=endJ(0.08,600)
# compare J at matched L by interpolation
Lg=np.linspace(1.6,7.0,20); j2=np.interp(Lg,L2,J2); j4=np.interp(Lg,L4,J4)
print("grid convergence N=300 vs 600 (alpha=0.08): max|dJ|=%.2e  rel=%.2e"%(np.max(np.abs(j2-j4)), np.max(np.abs(j2-j4))/np.max(np.abs(j4))))
