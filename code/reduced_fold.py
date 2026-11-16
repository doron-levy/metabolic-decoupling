import numpy as np

# --- bistable consumer phenotype: m^3 - m - c = 0, c = beta*(D - theta) ---
# oxidative branch (continue from m=+ at small separation); wells at m=+-1, local fold at c=-+2/(3 sqrt3)
xi_l, xi_r = 1.0, 0.4        # lactate attraction range, short-range steric repulsion range
a0, r0     = 2.0, 2.5        # attraction, repulsion amplitudes
theta, beta = 0.5, 1.5       # phenotype threshold, supply-to-bias gain
S0         = 1.3             # held lactate supply amplitude

def m_star_branch(Dgrid):
    # follow the oxidative (upper) root by continuity as D decreases
    m=np.empty_like(Dgrid); prev=1.0
    for i,D in enumerate(Dgrid):
        c=beta*(D-theta)
        roots=np.roots([1,0,-1,-c])            # m^3 - m - c
        roots=roots[np.abs(roots.imag)<1e-9].real
        prev=roots[np.argmin(np.abs(roots-prev))]
        m[i]=prev
    return m

fo=lambda m: np.clip(0.5*(1+m),0,1)

def build_G(N=1600, dmax=6.0, mono=False, S=S0):
    D2=np.linspace(dmax,0,N)                    # decreasing separation order for continuation
    Dvals=S*np.exp(-np.linspace(dmax,0,N)/xi_l)
    if mono:
        m=np.tanh(3.0*beta*(Dvals-theta))       # smooth monostable phenotype
    else:
        m=m_star_branch(Dvals)                  # bistable branch
    Delta=np.linspace(dmax,0,N)
    G = a0*fo(m)*np.exp(-Delta/xi_l) - r0*np.exp(-Delta/xi_r)
    idx=np.argsort(Delta)
    return Delta[idx], G[idx], m[idx]

D_b,G_b,m_b = build_G(mono=False)
D_m,G_m,m_m = build_G(mono=True)
np.savez("Gcurves.npz",D_b=D_b,G_b=G_b,m_b=m_b,D_m=D_m,G_m=G_m,m_m=m_m)

def Gfun(D, Dg, Gg): return np.interp(D, Dg, Gg)

# --- static fold: fixed points of alpha*Delta = G(Delta), held supply ---
def alpha_star(Dg,Gg):
    m=(Dg>0.05); return np.max(Gg[m]/Dg[m])     # tangency of line alpha*Delta to the hump
astar_b=alpha_star(D_b,G_b); astar_m=alpha_star(D_m,G_m)
print("alpha* (bistable)=%.4f   alpha* (monostable)=%.4f"%(astar_b,astar_m))

def stable_fp(alpha,Dg,Gg):
    f=Gg-alpha*Dg; s=np.where(np.diff(np.sign(f)))[0]   # zero crossings
    if len(s)==0: return np.nan
    # stable coordinated fp = smallest Delta where G crosses alpha*Delta from above (rising edge)
    return Dg[s[0]]

# --- dynamic decoupling time ---
def t_decouple(alpha, Dg, Gg, dilute=False, D0=0.25, Desc=5.0, dt=2e-3, Tmax=4000):
    D=D0; t=0.0
    while t<Tmax:
        clos=Gfun(D,Dg,Gg)*(np.exp(-alpha*t) if dilute else 1.0)
        D+= dt*(alpha*D - clos)
        t+=dt
        if D>Desc: return t
        if D<1e-3: D=1e-3
    return np.nan

if __name__=="__main__":
    # held-supply sharp fold: t_dec vs (alpha-alpha*) just above threshold
    eps=np.array([0.004,0.008,0.015,0.03,0.06,0.12,0.25])
    th=[t_decouple(astar_b+e, D_b,G_b, dilute=False) for e in eps]
    th=np.array(th)
    sl_hold=np.polyfit(np.log(eps[np.isfinite(th)]), np.log(th[np.isfinite(th)]),1)[0]
    print("HELD  t_dec vs (alpha-alpha*): fitted exponent = %.3f  (saddle-node predicts -0.5)"%sl_hold)
    for e,t in zip(eps,th): print("   d_alpha=%.3f  t_dec=%.2f"%(e,t))

    # dilution soft slow-passage: t_dec vs alpha (small alpha), expect ~1/alpha
    als=np.array([0.05,0.08,0.12,0.2,0.35,0.6])
    td=np.array([t_decouple(a, D_b,G_b, dilute=True) for a in als])
    sl_dil=np.polyfit(np.log(als[np.isfinite(td)]), np.log(td[np.isfinite(td)]),1)[0]
    print("DILUTE t_dec vs alpha: fitted exponent = %.3f  (slow passage predicts -1)"%sl_dil)
    for a,t in zip(als,td): print("   alpha=%.3f  t_dec=%.2f"%(a,t))
    np.savez("folddata.npz",eps=eps,th=th,als=als,td=td,astar_b=astar_b,astar_m=astar_m,sl_hold=sl_hold,sl_dil=sl_dil)
