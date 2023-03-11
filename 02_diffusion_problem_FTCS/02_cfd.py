#! /usr/bin/env python
# -*- coding:utf-8 -*-

################################################################
#
# Class 02: Solve the 1D-diffusion equation using the Forward-Time Central-Space scheme
# Book: "The minimum you need to know to write your own CFD solver"
# Author: Pedro Stefanin Volpiani
# Online course: https://youtu.be/tB7i0YuxklA
#
################################################################

#===============================================================
# Some libraries
#===============================================================
import matplotlib.pyplot as plt
import numpy as np

# To make your graphics nicer
plt.rc('font', family='serif', size=16)
plt.rc('lines', linewidth=1.5)
plt.rc('legend', fontsize=12)

#===============================================================
# Define parameters
#===============================================================
alpha = 0.01                      # Diffusivity
sigma = 0.4                       # Stability condition
Nx = 101;                         # Number of grid points
xmax = 2.;                        # Domain limit to the right
xmin = -2.;                       # Domain limit to the left
Lx = xmax-xmin                    # Domain size
dx = Lx/(Nx-1)                    # Mesh size
x = np.linspace(xmin,xmax,Nx)     # Discretized mesh
dt = sigma*dx**2/alpha            # Time step
t_end = 5.                        # Final time
Nt = int(t_end/dt)                # Number of iterations
U = np.zeros((Nt+1,Nx))           # u^n_i
U[0,:] = np.exp(-0.5*(x/0.4)**2)  # Initial solution

#===============================================================
# Temporal loop
#===============================================================
for n in range (0,Nt):
  
  sigma = alpha*dt/(dx*dx)

  for i in range (1,Nx-1): # Interior points
      U[n+1,i] = U[n,i] + sigma*(U[n,i+1]-2*U[n,i]+U[n,i-1]);
  U[n+1,0] = 0;  # BC left
  U[n+1,-1] = 0; # BC right
  
  # Plot solution
  if (n==0): fig, ax = plt.subplots(figsize=(5.5,4))
  plt.clf()
  plt.plot(x,U[n+1,:])
  plt.scatter(x,U[0,:], marker='o', facecolors='white', color='k')
  plt.gca().legend(('Numerical result ($\sigma$='+str(sigma)+')','Initial condition'))
  plt.axis([xmin, xmax, 0, 1.4])
  plt.title('t='+str(round(dt*(n+1),3)),fontsize=16)
  plt.xlabel('x',fontsize=18)
  plt.ylabel('u',fontsize=18)
  plt.tight_layout()
  plt.draw(); plt.pause(0.001)

plt.show()
#fig.savefig("figure.pdf", dpi=300)
