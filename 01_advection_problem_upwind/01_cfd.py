#! /usr/bin/env python
# -*- coding:utf-8 -*-

################################################################
#
# Class 01: Solve the 1D-advection equation using the upwind scheme
# Book: "The minimum you need to know to write your own CFD solver"
# Author: Pedro Stefanin Volpiani
# Online course: https://youtu.be/QcYz67uORLM
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
Nx = 101                          # Number of grid points
xmax = 2.                         # Domain limit to the right
xmin = -2.                        # Domain limit to the left
Lx = xmax-xmin                    # Domain size
dx = Lx/(Nx-1)                    # Mesh size
x = np.linspace(xmin,xmax,Nx)     # Discretized mesh
dt = 0.04                         # Time step
t_end = 5.                        # Final time
Nt = int(t_end/dt)                # Number of iterations
a = 0.8                           # Advection speed
CFL = a*dt/dx                     # CFL number
U = np.zeros((Nt+1,Nx))           # u^n_i
U[0,:] = np.exp(-0.5*(x/0.4)**2)  # Initial solution
Uex = U[0,:]                      # Exact solution

#===============================================================
# Solve equation using the upwind scheme
#===============================================================
for n in range (0,Nt):

  if (a>0.):
      for i in range (1,Nx):
          U[n+1,i] = U[n,i] - CFL*(U[n,i]-U[n,i-1]);
      U[n+1,0] = U[n+1,Nx-1];
  else:
      for i in range (0,Nx-1):
          U[n+1,i] = U[n,i] - CFL*(U[n,i+1]-U[n,i]);
      U[n+1,Nx-1] = U[n+1,0];
      
#===============================================================
# Compute exact solution
#===============================================================
  d = a*(n+1)*dt
  Uex = np.exp(-0.5*(np.mod(x-d+xmax,4)-xmax)**2/0.4**2)
  errL1 = U - Uex
  errL2 = np.linalg.norm(errL1)
  
#===============================================================
# Plot solution
#===============================================================
  if (n==0): fig, ax = plt.subplots(figsize=(5.5,4))
  plt.clf()
  plt.plot(x,U[n+1,:])
  plt.scatter(x,Uex, marker='o', facecolors='white', color='k')
  plt.gca().legend(('Upwind scheme (CFL='+str(CFL)+')','Exact solution'))
  plt.axis([xmin, xmax, 0, 1.4])
  plt.title('t='+str(round(dt*(n+1),3)),fontsize=16)
  plt.xlabel('x',fontsize=18)
  plt.ylabel('u',fontsize=18)
  plt.tight_layout()
  plt.draw(); plt.pause(0.001)

plt.show()
#fig.savefig("figure.pdf", dpi=300)
#print 'Error L2 = ',errL2
