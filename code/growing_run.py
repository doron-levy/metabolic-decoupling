import numpy as np
from scipy.linalg import solve_banded

W, N = 8.0, 500
Dl, lam_l = 1.0, 1.0
xi_r, mu_r = 0.4, 2.2
xi_s, mu_s = 0.6, 1.6
chi        = 1.6
P, R, Kl   = 2.0, 1.0, 0.5
theta,beta = 0.5, 1.6
tau_m      = 0.25            # fast bistable phenotype relaxation
Dcell      = 4e-3
Ppack      = 3.0
z  = np.linspace(-W, W, N); dx = z[1]-z[0]
fo = lambda m: np.clip(0.5*(1+m),0,1)
gL = lambda l: l/(l+Kl)

def tri(diag,lo,up,rhs):
    ab=np.zeros((3,N)); ab[0,1:]=up[:-1]; ab[1,:]=diag; ab[2,:-1]=lo[1:]
    return solve_banded((1,1),ab,rhs)
def cumint(f):
    C=np.zeros(N+1); C[1:]=np.cumsum(f)*dx; return C[1:]
def nl_grad(src,xi):
    CU=cumint(src); ix=int(round(xi/dx))
    Cp=np.empty(N);Cm=np.empty(N)
    Cp[:-ix]=CU[ix:];Cp[-ix:]=CU[-1]; Cm[ix:]=CU[:-ix];Cm[:ix]=CU[0]
    return Cp-2*CU+Cm
def lactate(u,w,m):
    c=Dl/dx**2; diag=(lam_l+R*fo(m)*w/Kl)+2*c; lo=-c*np.ones(N);up=-c*np.ones(N)
    diag[0]-=c;diag[-1]-=c; return tri(diag,lo,up,P*u)
def minmod(a,b): return np.where(a*b>0,np.sign(a)*np.minimum(np.abs(a),np.abs(b)),0.0)
def lap(f): return (np.roll(f,-1)-2*f+np.roll(f,1))/dx**2
def adv(rho,V):
    r=np.concatenate([[0.,0.],rho,[0.,0.]]); d=r[1:]-r[:-1]
    sl=minmod(d[1:],d[:-1]); rc=r[2:-2]; slc=sl[1:-1]
    Vf=0.5*(V[:-1]+V[1:])
    rL=rc[:-1]+0.5*slc[:-1]; rR=rc[1:]-0.5*slc[1:]
    Fi=np.where(Vf>0,Vf*rL,Vf*rR)
    F=np.zeros(N+1); F[1:-1]=Fi
    F[0]=V[0]*rc[0] if V[0]<0 else 0.0
    F[-1]=V[-1]*rc[-1] if V[-1]>0 else 0.0
    return -(F[1:]-F[:-1])/dx
def vel(u,w,l,m,alpha):
    dl=np.gradient(l,dx); pk=np.clip(1-(u+w)/Ppack,0,1)
    Vu=alpha*z + pk*mu_s*nl_grad(u,xi_s) - mu_r*nl_grad(w,xi_r)
    Vw=alpha*z + pk*mu_s*nl_grad(w,xi_s) - mu_r*nl_grad(u,xi_r) + pk*chi*fo(m)*dl
    return Vu,Vw
def rates(u,w,m,alpha,hold):
    l=lactate(u,w,m); Vu,Vw=vel(u,w,l,m,alpha)
    du=adv(u,Vu)+Dcell*lap(u); dw=adv(w,Vw)+Dcell*lap(w)
    dm=(m - m**3 + beta*(gL(l)-theta))/tau_m
    if hold: du+=alpha*u; dw+=alpha*w
    return du,dw,dm
def step(u,w,m,alpha,hold,dt):
    du,dw,dm=rates(u,w,m,alpha,hold)
    u1=np.clip(u+dt*du,0,None); w1=np.clip(w+dt*dw,0,None); m1=np.clip(m+dt*dm,-1.5,1.5)
    du2,dw2,dm2=rates(u1,w1,m1,alpha,hold)
    u=np.clip(u+0.5*dt*(du+du2),0,None); w=np.clip(w+0.5*dt*(dw+dw2),0,None)
    m=np.clip(m+0.5*dt*(dm+dm2),-1.5,1.5)
    return u,w,m
def cen(rho):
    M=np.trapezoid(rho,z); return (np.trapezoid(z*rho,z)/M if M>1e-9 else np.nan)
def recenter(u,w,m,shift):
    if shift==0: return u,w,m
    u=np.roll(u,-shift); w=np.roll(w,-shift); m=np.roll(m,-shift)
    if shift>0:
        u[-shift:]=0; w[-shift:]=0; m[-shift:]=-1.1
    else:
        u[:-shift]=0; w[:-shift]=0; m[:-shift]=-1.1
    return u,w,m

def run(alpha,hold=False,D0=0.8,Tmax=60.0,dt=1.5e-3,rec=0.5):
    u=np.exp(-(z+D0/2)**2/(2*0.4**2)); u/=np.trapezoid(u,z)
    w=np.exp(-(z-D0/2)**2/(2*0.4**2)); w/=np.trapezoid(w,z)
    m=np.ones(N); ts=[];sp=[];t=0.0;k=0;nid=max(1,int(rec/dt))
    while t<Tmax:
        u,w,m=step(u,w,m,alpha,hold,dt)
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(w)):
            ts.append(t);sp.append(np.inf); break
        Mu=np.trapezoid(u,z);Mw=np.trapezoid(w,z)
        Xu=np.trapezoid(z*u,z)/Mu; Xw=np.trapezoid(z*w,z)/Mw
        s_=Xw-Xu
        Xc=(Mu*Xu+Mw*Xw)/(Mu+Mw)
        u,w,m=recenter(u,w,m,int(round(Xc/dx)))
        if k%nid==0:
            ts.append(t);sp.append(s_)
            if (not np.isfinite(s_)) or s_>2.5 or Mu<0.4 or Mw<0.4: break
        t+=dt;k+=1
    lf = lactate(u,w,m) if np.all(np.isfinite(u+w+m)) else np.zeros(N)
    return np.array(ts),np.array(sp),(u,w,m,lf)

if __name__=="__main__":
    # (a) held-supply separation trajectories (Figure 6a)
    hd={}
    for al in [0.02,0.04,0.06,0.08,0.12]:
        t,sp,_=run(al,hold=True,Tmax=40.0)
        hd["t%.2f"%al]=t; hd["s%.2f"%al]=sp
    np.savez("held_sweep.npz",**hd)

    # (b,c) snapshots: coordinated (held) vs decoupled (dilution) at the same growth rate
    _,_,(uC,wC,mC,lC)=run(0.06,hold=True, Tmax=40.0)
    _,_,(uD,wD,mD,lD)=run(0.06,hold=False,Tmax=90.0)
    np.savez("snapshots.npz",z=z,uC=uC,wC=wC,mC=mC,lC=lC,uD=uD,wD=wD,mD=mD,lD=lD)

    # (d) decoupling time vs growth rate: held (sharp threshold) vs dilution (soft) (Figure 6d)
    hk=[];hv=[];dk=[];dv=[]
    for al in [0.08,0.10,0.13,0.16,0.20]:
        t,sp,_=run(al,hold=True, Tmax=90.0); hk.append(al); hv.append(t[-1])
    for al in [0.06,0.08,0.11,0.15,0.20]:
        t,sp,_=run(al,hold=False,Tmax=90.0); dk.append(al); dv.append(t[-1])
    np.savez("tdec.npz",hk=np.array(hk),hv=np.array(hv),dk=np.array(dk),dv=np.array(dv))
    print("held tdec:",np.round(hv,1)); print("dilu tdec:",np.round(dv,1))
    print("held_sweep.npz, snapshots.npz, tdec.npz written")
