import numpy as np
from scipy.linalg import solve_banded

# length unit = lactate range xi_l = 1 ; time unit = phenotype relaxation tau = 1
Dl, lam_l = 1.0, 1.0          # -> xi_l = 1
xi_c      = 0.5               # oxygen penetration depth (in xi_l)
kappa     = 0.02             # phenotype interface stiffness
P, R, Kl  = 2.0, 1.0, 0.5     # lactate production, oxidative consumption, half-saturation
beta_o, beta_g = 3.0, 2.0     # oxidative drive (O2 & lactate), glycolytic drive (hypoxia)
rho0, L0  = 1.0, 1.5

fo = lambda phi: 0.5*(1.0-phi)   # oxidative fraction  (s=-1 oxidative, matching the paper)
fg = lambda phi: 0.5*(1.0+phi)   # glycolytic fraction (s=+1 glycolytic)
gl = lambda l: l/(l+Kl)

def oxygen(xh,L):
    b=L/xi_c
    # cosh(xh b)/cosh(b) computed stably for large b (o decays from rim xh=1)
    return np.exp((xh-1.0)*b)*(1.0+np.exp(-2.0*xh*b))/(1.0+np.exp(-2.0*b))

def tri(diag, lo, up, rhs):
    N=len(diag); ab=np.zeros((3,N)); ab[0,1:]=up[:-1]; ab[1,:]=diag; ab[2,:-1]=lo[1:]
    return solve_banded((1,1), ab, rhs)

def solve_lactate(phi,o,L,rho,N,dx):
    c=Dl/(L*L*dx*dx)
    diag=(lam_l+R*fo(phi)*o)+2*c; lo=-c*np.ones(N); up=-c*np.ones(N)
    diag[0]-=c; diag[-1]-=c        # Neumann
    return tri(diag,lo,up,P*fg(phi)*rho)

def phi_target(l,o):
    return np.tanh(beta_g*(1.0-o) - beta_o*o*gl(l))   # +1 glycolytic where hypoxic, -1 oxidative

def step_phi(phi,phistar,L,dt,dx):
    c=kappa/(L*L*dx*dx)*dt; N=len(phi)
    diag=(1+dt+2*c)*np.ones(N); lo=-c*np.ones(N); up=-c*np.ones(N)
    diag[0]-=c; diag[-1]-=c
    return tri(diag,lo,up, phi + dt*phistar)          # implicit diffusion+decay, target source

def observ(phi,l,o,L,dx):
    Mox=np.trapezoid(np.clip(fo(phi),0,1),dx=dx)
    J  =np.trapezoid(fo(phi)*o*gl(l),dx=dx)*L
    return Mox,J

def relax_static(L,N=400,iters=40000,tol=1e-9,seed=-0.9):
    dx=1.0/(N-1); xh=np.linspace(0,1,N); o=oxygen(xh,L); rho=rho0*L0/L
    phi=seed*np.ones(N); dt=5e-3
    for k in range(iters):
        l=solve_lactate(phi,o,L,rho,N,dx); ps=phi_target(l,o); pn=step_phi(phi,ps,L,dt,dx)
        if np.max(np.abs(pn-phi))/dt<tol: phi=pn; break
        phi=pn
    l=solve_lactate(phi,o,L,rho,N,dx)
    return phi,l,o,observ(phi,l,o,L,dx)

def grow(alpha,Tmax,N=400,dt=2e-3,seedL=L0):
    dx=1.0/(N-1); xh=np.linspace(0,1,N)
    phi,l,o,_=relax_static(seedL,N=N)
    ts,Ls,Js,Ms=[],[],[],[]; t=0.0; nst=int(Tmax/dt)
    for k in range(nst):
        L=seedL*np.exp(alpha*t); rho=rho0*L0/L; o=oxygen(xh,L)
        l=solve_lactate(phi,o,L,rho,N,dx); ps=phi_target(l,o); phi=step_phi(phi,ps,L,dt,dx)
        if k%25==0:
            M,J=observ(phi,l,o,L,dx); ts.append(t);Ls.append(L);Js.append(J);Ms.append(M)
        t+=dt
    return [np.array(z) for z in (ts,Ls,Js,Ms)]

if __name__=="__main__":
    Ls=np.linspace(1.0,7.0,31); Js=[];Ms=[]
    for L in Ls:
        _,_,_,(M,J)=relax_static(L); Js.append(J);Ms.append(M)
    Js=np.array(Js);Ms=np.array(Ms)
    np.savez("static.npz",Ls=Ls,Js=Js,Ms=Ms)
    J0=Js.max()
    for L,J,M in zip(Ls,Js,Ms): print(f"  L={L:4.2f}  J={J:7.4f}  J/J0={J/J0:5.2f}  Mox={M:5.3f}")
    # half-max threshold
    half=0.5*J0
    Lc=np.interp(-half,-Js[np.argmax(Js):],Ls[np.argmax(Js):])
    print(f"\n J0={J0:.4f}  L*(half-max shuttle) approx = {Lc:.2f}")
