import numpy as np
from scipy.linalg import solve_banded

# ---- full bistable two-population operator; measure closing rate G(Delta) by centroid velocities ----
Lx, N = 24.0, 1200
Dl, lam_l = 1.0, 1.0          # lactate: xi_l = 1
xi_r      = 0.4               # steric repulsion range (short: the metabolic "run")
sigma, Xc = 0.5, 12.0         # cluster width, domain centre
P, R, Kl  = 2.0, 1.0, 0.5     # lactate production (by u), consumption (by oxidative w), half-sat
chi       = 1.6              # chemotaxis of w up lactate gradient (gated by oxidative phenotype)
mu_r      = 2.2              # steric repulsion strength
theta,beta= 0.5, 1.6         # bistable phenotype threshold and gain
a_ph,b_ph = 1.0, 1.0

def grid(N):
    x=np.linspace(0,Lx,N); return x, x[1]-x[0]

def gaussian(x,X0):
    g=np.exp(-(x-X0)**2/(2*sigma**2)); return g/np.trapezoid(g,x)   # unit mass

fo=lambda m: np.clip(0.5*(1+m),0,1)
gL=lambda l: l/(l+Kl)

def tri_solve(diag,lo,up,rhs):
    Nn=len(diag); ab=np.zeros((3,Nn)); ab[0,1:]=up[:-1]; ab[1,:]=diag; ab[2,:-1]=lo[1:]
    return solve_banded((1,1),ab,rhs)

def lactate(u,w,m,x,dx):
    # (-Dl d2 + lam_l + R fo(m) w /Kl) l = P u   (consumption linearised in l), Neumann
    c=Dl/dx**2
    diag=(lam_l + R*fo(m)*w/Kl)+2*c; lo=-c*np.ones(len(u)); up=-c*np.ones(len(u))
    diag[0]-=c; diag[-1]-=c
    return tri_solve(diag,lo,up,P*u)

def m_branch(ldrive, m_prev):
    # oxidative branch of a m - b m^3 + beta(gL(l)-theta)=0, pointwise root nearest previous
    cc=beta*(gL(ldrive)-theta); m=np.empty_like(cc)
    for i in range(len(cc)):
        r=np.roots([b_ph,0,-a_ph,-cc[i]]); r=r[np.abs(r.imag)<1e-9].real
        m[i]=r[np.argmin(np.abs(r-m_prev[i]))]
    return m

def m_mono(ldrive):
    return np.tanh(3.0*beta*(gL(ldrive)-theta))

def steric_vel(src, x, dx):
    # v_rep(x) = -mu_r [ CU(x+xi_r) - 2 CU(x) + CU(x-xi_r) ],  CU cumulative mass of source
    CU=np.concatenate([[0],np.cumsum(src)*dx]); CU=CU[1:]
    ir=int(round(xi_r/dx))
    Cp=np.empty_like(CU); Cm=np.empty_like(CU)
    Cp[:-ir]=CU[ir:]; Cp[-ir:]=CU[-1]
    Cm[ir:]=CU[:-ir]; Cm[:ir]=CU[0]
    return -mu_r*(Cp-2*CU+Cm)

def measure_G(Delta,x,dx,mono=False,m_seed=None):
    u=gaussian(x,Xc-Delta/2); w=gaussian(x,Xc+Delta/2)
    m = m_seed.copy() if m_seed is not None else np.ones(len(x))
    for _ in range(60):
        l=lactate(u,w,m,x,dx)
        m_new = m_mono(l) if mono else m_branch(l,m)
        if np.max(np.abs(m_new-m))<1e-8: m=m_new; break
        m=m_new
    l=lactate(u,w,m,x,dx)
    dl=np.gradient(l,dx)
    v_att = chi*fo(m)*dl                 # chemotaxis of w up lactate gradient
    v_rep_w = steric_vel(u,x,dx)         # w repelled by u
    v_rep_u = steric_vel(w,x,dx)         # u repelled by w
    Vw = np.trapezoid(w*(v_att+v_rep_w),x)/np.trapezoid(w,x)
    Vu = np.trapezoid(u*(v_rep_u),x)/np.trapezoid(u,x)
    G = Vu - Vw                          # closing rate: >0 clusters approach
    return G, m

def sweep_G(N=N, mono=False):
    x,dx=grid(N); Ds=np.arange(0.25,6.01,0.10); G=np.empty_like(Ds); m=np.ones(N)
    for i,D in enumerate(Ds):
        G[i],m=measure_G(D,x,dx,mono=mono,m_seed=m)
    return Ds,G

if __name__=="__main__":
    Db,Gb=sweep_G(mono=False); Dm,Gm=sweep_G(mono=True)
    np.savez("Gmeasured.npz",Db=Db,Gb=Gb,Dm=Dm,Gm=Gm)
    zc=Db[np.where(np.diff(np.sign(Gb)))[0]]
    print("BISTABLE  G(contact)=%.4f  peak=%.4f at D=%.2f  zero(s) at D=%s"%(Gb[0],Gb.max(),Db[np.argmax(Gb)],np.round(zc,2)))
    # locate the cliff (largest negative jump)
    dG=np.diff(Gb); ic=np.argmin(dG)
    print("  sharpest drop between D=%.2f and %.2f : dG=%.4f (the bistable cliff)"%(Db[ic],Db[ic+1],dG[ic]))
    print("MONOSTABLE G(contact)=%.4f  peak=%.4f  min dG=%.4f (smooth, no cliff)"%(Gm[0],Gm.max(),np.diff(Gm).min()))
    # grid convergence of measured G
    D2,G2=sweep_G(N=2400,mono=False)
    err=np.max(np.abs(np.interp(Db,D2,G2)-Gb))
    print("grid convergence N=1200 vs 2400: max|dG|=%.2e"%err)
