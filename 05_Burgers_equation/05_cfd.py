#! /usr/bin/env python
# -*- coding:utf-8 -*-

################################################################
#
# Class 05: Solve the 1D-Inviscid Burguers' equation using the upwind scheme
# Book: "The minimum you need to know to write your own CFD solver"
# Author: Pedro Stefanin Volpiani
# Online course: https://youtu.be/gsIcY1lUOcY
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
plt.rc('legend', fontsize=11)

#===============================================================
# Some definitions
#===============================================================
scheme = 2                        # Chose your scheme
Nx = 101;                         # Number of grid points
xmax = 2.;                        # Domain limit to the right
xmin = -2.;                       # Domain limit to the left
dx = (xmax-xmin)/(Nx-1)           # Mesh size
x = np.arange(xmin,xmax,dx)       # Discretized mesh

dt = 0.04                         # Time step
t = 0.                            # Initial time
t_end = 2.                        # Final time

# Exact solution at t = 0
Uex0 = (x<-1)*0+(x<0)*(x>-1)*(1+x)+(x<1)*(x>0)*(1-x)+(x>1)*0
# Exact solution at t = 1
Uex1 = (x<-1)*0+(x<1)*(x>-1)*(0.5+0.5*x)+(x<1)*(x>1)*1+(x>1)*0
# Exact solution at t = 2
Uex2 = (x<-1)*0+(x<1.5)*(x>-1)*(0.32+0.32*x)+(x>1)*0

U = Uex0.copy()                   # Initial solution

# Temporal loop
while (t <= t_end):

  dt = min(dt,dx / max(abs(U)))
  sigma = dt/dx
  t = t+dt
  
  # Solve equation using upwind scheme
  if (scheme == 1):
    
      Un = U
      Um = np.roll(Un,1)
      Up = np.roll(Un,-1)
      U = Un - 0.5 * sigma * (Un + abs(Un)) * (Un-Um) - 0.5 * sigma * (Un - abs(Un)) * (Up - Un)

  # Solve equation using the corrected upwind scheme
  if (scheme == 2):
  
      Un = U
      Um = np.roll(Un,1)
      Up = np.roll(Un,-1)
      a_plus = 0.5 * (Up + Un)
      a_minus = 0.5 * (Un + Um)
      
      U = Un - 0.5 * sigma * (a_minus + abs(a_minus)) * (Un-Um) - 0.5 * sigma * (a_plus - abs(a_plus)) * (Up-Un)
      
  # Plot solution
  if (t==dt): fig, ax = plt.subplots(figsize=(5.5,4))
  if (abs(t - 1) < 1e-9): plt.plot(x,U)
  if (abs(t - 2) < 1e-9): plt.plot(x,U)
  plt.scatter(x,Uex0, marker='o', facecolors='white', color='k')
  plt.scatter(x,Uex1, marker='o', facecolors='white', color='C0')
  plt.scatter(x,Uex2, marker='o', facecolors='white', color='C1')
  plt.gca().legend(('Modified scheme (t=1s)','Modified scheme (t=2s)','Initial solution','Exact solution (t=1s)','Exact solution (t=2s)'))
  plt.axis([xmin, xmax, 0, 1.5])
  plt.xlabel('x',fontsize=18)
  plt.ylabel('u',fontsize=18)
  plt.tight_layout()

plt.show()
#fig.savefig("ex1_burgers_schema2.pdf", dpi=300)


