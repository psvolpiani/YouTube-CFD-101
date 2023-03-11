#! /usr/bin/env python
# -*- coding:utf-8 -*-

################################################################
#
# Class 08: Solve the 1D-Euler system of equations using the MUSCL scheme
# Book: "The minimum you need to know to write your own CFD solver"
# Author: Pedro Stefanin Volpiani
# Online course: https://youtu.be/XPmJ1FLcMqA
#
################################################################

import numpy as np
import matplotlib.pyplot as plt

from numpy import *
from matplotlib import rc

rc('font', family='serif', size=14)
rc('lines', linewidth=1.5)

################################################################
#             Solve the 1D-Euler system of equations
#                    using the MUSCL scheme
#
#             dq_i/dt + df_i/dx = 0, for x \in [a,b]
#
# This code solves the Sod's shock tube problem (IC=1)
#
#    t=0                          t=tEnd
#    Density                      Density
#    **********|                  *******\
#              |                          \
#              |                           \
#              |                            ****|
#              |                                |
#              |                                ****|
#              **********                           ********
#
# Domain cells (I{i}) reference:
#
#              |           |   u(i)    |           |
#              |  u(i-1)   |___________|           |
#              |___________|           |   u(i+1)  |
#              |           |           |___________|
#           ...|-----0-----|-----0-----|-----0-----|...
#              |    i-1    |     i     |    i+1    |
#              |-         +|-         +|-         +|
#            i-3/2       i-1/2       i+1/2       i+3/2
#
################################################################

#===============================================================
# We need a function to compute the primitive variables
#===============================================================
def func_cons2prim(q,gamma):
    # Primitive variables
    r=q[0];
    u=q[1]/r;
    E=q[2]/r;
    p=(gamma-1.)*r*(E-0.5*u**2);
    
    return (r,u,p)
    
#===============================================================
# We need a function to compute the flux vector
#===============================================================
def func_flux(q,gamma):
    # Primitive variables
    r=q[0];
    u=q[1]/r;
    E=q[2]/r;
    p=(gamma-1.)*r*(E-0.5*u**2);
    
    # Flux vector
    F0 = np.array(r*u)
    F1 = np.array(r*u**2+p)
    F2 = np.array(u*(r*E+p))
    flux=np.array([ F0, F1, F2 ]);
    
    return (flux)

#===============================================================
# Limiter function
#===============================================================
def psi(r):
    # Limiter functions
    beta = 1.0
    psi_r = np.maximum(0, np.maximum(np.minimum(beta*r, 1.), np.minimum(r, beta)))

    return psi_r

#===============================================================
# Function for the MUSCL reconstruction
#===============================================================
def flux_muscl(q,dx,gamma,a,nx):
    
    # Compute and limit slopes
    dqL=np.zeros((3,nx-1)); dqR=np.zeros((3,nx-1));
    qL =np.zeros((3,nx-1)); qR =np.zeros((3,nx-1));

    for i in range (0,3):
        for j in range (1,nx-2): # for all internal faces
                # Find dq_j
                
                num = (q[i,j] - q[i,j-1])
                den = (q[i,j+1] - q[i,j])
                if (abs(num) < 1E-8) : num=0.; den=1.
                elif (num > 1E-8 and abs(den) < 1E-8): num= 1.; den=1.
                elif (num <-1E-8 and abs(den) < 1E-8): num=-1.; den=1.
                dqL[i,j] = psi(num/den)
                
                num = (q[i,j+1] - q[i,j])
                den = (q[i,j+2] - q[i,j+1])
                if (abs(num) < 1E-8) : num=0.; den=1.
                elif (num > 1E-8 and abs(den) < 1E-8): num= 1.; den=1.
                elif (num <-1E-8 and abs(den) < 1E-8): num=-1.; den=1.
                dqR[i,j] = psi(num/den)
 
    # Left and Right extrapolated q-values at the boundary j+1/2
    for j in range (1,nx-2): # for the domain cells
        qL[:,j] = q[:, j ] + 0.5*dqL[:,j]*(q[:,j+1] - q[:, j ]); # q_{j+1/2}^{L} from j
        qR[:,j] = q[:,j+1] - 0.5*dqR[:,j]*(q[:,j+2] - q[:,j+1]); # q_{j+1/2}^{R} from j+1
    
    # Flux contribution of the LEFT MOST FACE: left face of cell j=1.
    qR[:,0]=q[:,1]-0.5*dqR[:,1]*(q[:,1+2] - q[:,1+1]);    qL[:,0] = qR[:,0];
    # Flux contribution of the RIGTH MOST FACE: right face of cell j=nx-2.
    qL[:,nx-2]=q[:,nx-2]+0.5*dqL[:,nx-2]*(q[:,nx-1] - q[:,nx-2]); qR[:,nx-2] = qL[:,nx-2];
    
    # Compute flux at j+1/2
    Phi=np.zeros((3,nx-1))
    for j in range (0,nx-1): # for the domain cells
        Phi[:,j] = flux_roe(qL[:,j],qR[:,j],gamma)
    
    dF = (Phi[:,1:-1]-Phi[:,0:-2])
    return(dF)
    
#===============================================================
# We need a function to compute the Roe flux
#===============================================================
def flux_roe(qL,qR,gamma):

    # Compute left state
    (rL,uL,pL) = func_cons2prim(qL,gamma)
    hL = gamma/(gamma-1)*pL/rL+0.5*uL**2;
    # Compute right state
    (rR,uR,pR) = func_cons2prim(qR,gamma)
    hR = gamma/(gamma-1)*pR/rR+0.5*uR**2;

    # Compute Roe averages
    R=sqrt(rR/rL);                                # R_{j+1/2}
    rmoy=R*rL;                                    # {hat rho}_{j+1/2}
    umoy=(R*uR+uL)/(R+1);                         # {hat U}_{j+1/2}
    hmoy=(R*hR+hL)/(R+1);                         # {hat H}_{j+1/2}
    amoy=sqrt((gamma-1.0)*(hmoy-0.5*umoy*umoy));  # {hat a}_{j+1/2}
    
    # Auxiliary variables used to compute P_{j+1/2}^{-1}
    alph1=(gamma-1)*umoy*umoy/(2*amoy*amoy);
    alph2=(gamma-1)/(amoy*amoy);

    # Compute vector (W_{j+1}-W_j)
    wdif = qR-qL;
    
    # Compute matrix P^{-1}_{j+1/2}
    Pinv = np.array([[0.5*(alph1+umoy/amoy), -0.5*(alph2*umoy+1/amoy),  alph2/2],
                    [1-alph1,                alph2*umoy,                -alph2 ],
                    [0.5*(alph1-umoy/amoy),  -0.5*(alph2*umoy-1/amoy),  alph2/2]]);
            
    # Compute matrix P_{j+1/2}
    P    = np.array([[ 1,              1,              1              ],
                    [umoy-amoy,        umoy,           umoy+amoy      ],
                    [hmoy-amoy*umoy,   0.5*umoy*umoy,  hmoy+amoy*umoy ]]);
    
    # Compute matrix Lambda_{j+1/2}
    lamb = np.array([[ abs(umoy-amoy),  0,              0                 ],
                    [0,                 abs(umoy),      0                 ],
                    [0,                 0,              abs(umoy+amoy)    ]]);
                  
    # Compute Roe matrix |A_{j+1/2}|
    A=np.dot(P,lamb)
    A=np.dot(A,Pinv)
    
    # Compute |A_{j+1/2}| (W_{j+1}-W_j)
    Phi=np.dot(A,wdif)
        
    # Compute Phi=(F(W_{j+1}+F(W_j))/2-|A_{j+1/2}| (W_{j+1}-W_j)/2
    FL = func_flux(qL,gamma);
    FR = func_flux(qR,gamma);
    Phi=0.5*(FL+FR)-0.5*Phi
    
    return (Phi)

#===============================================================
# Parameters
#===============================================================
CFL    = 0.50               # Courant Number
gamma  = 1.4                # Ratio of specific heats
tEnd = 0.20                 # Final time
ncells = 400                # Number of cells
x_ini =0.; x_fin = 1.       # Limits of computational domain
dx = (x_fin-x_ini)/ncells   # Step size
nx = ncells+1               # Number of points
x = np.linspace(x_ini+dx/2.,x_fin,nx) # Mesh

#===============================================================
# Initial conditions
#===============================================================
r0 = np.zeros(nx)
u0 = np.zeros(nx)
p0 = np.zeros(nx)
halfcells = int(ncells/2)

print ("Configuration 1, Sod's Problem")
p0[:halfcells] = 1.0  ; p0[halfcells:] = 0.1;
u0[:halfcells] = 0.0  ; u0[halfcells:] = 0.0;
r0[:halfcells] = 1.0  ; r0[halfcells:] = 0.125;

E0 = p0/((gamma-1.)*r0)+0.5*u0**2 # Total Energy density
a0 = sqrt(gamma*p0/r0)            # Speed of sound
q  = np.array([r0,r0*u0,r0*E0])   # Vector of conserved variables

#===============================================================
# Solver loop
#===============================================================
t  = 0
it = 0
a  = a0
dt=CFL*dx/max(abs(u0)+a0)         # Using the system's largest eigenvalue

while t < tEnd:

    q0 = q.copy();
    dF = flux_muscl(q0,dx,gamma,a,nx);
    
    q[:,1:-2] = q0[:,1:-2]-dt/dx*dF;
    q[:,0]=q0[:,0]; q[:,-1]=q0[:,-1]; # Dirichlet BCs
    
    # Compute primary variables
    rho=q[0];
    u=q[1]/rho;
    E=q[2]/rho;
    p=(gamma-1.)*rho*(E-0.5*u**2);
    a=sqrt(gamma*p/rho);
    if min(p)<0: print ('negative pressure found!')
    
    # Update/correct time step
    dt=CFL*dx/max(abs(u)+a);
    
    # Update time and iteration counter
    t=t+dt; it=it+1;
    
    # Plot solution
    if it%40 == 0:
        print (it)
        fig,axes = plt.subplots(nrows=4, ncols=1)
        plt.subplot(4, 1, 1)
        #plt.title('MUSCL scheme')
        plt.plot(x, rho, 'k-')
        plt.ylabel('$rho$',fontsize=16)
        plt.tick_params(axis='x',bottom=False,labelbottom=False)
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(x, u, 'r-')
        plt.ylabel('$U$',fontsize=16)
        plt.tick_params(axis='x',bottom=False,labelbottom=False)
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(x, p, 'b-')
        plt.ylabel('$p$',fontsize=16)
        plt.tick_params(axis='x',bottom=False,labelbottom=False)
        plt.grid(True)
    
        plt.subplot(4, 1, 4)
        plt.plot(x, E, 'g-')
        plt.ylabel('$E$',fontsize=16)
        plt.grid(True)
        plt.xlim(x_ini,x_fin)
        plt.xlabel('x',fontsize=16)
        plt.subplots_adjust(left=0.2)
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(top=0.95)
        plt.show()
        #fig.savefig("fig_Sod_MUSCL_it"+str(it)+".pdf", dpi=300)
