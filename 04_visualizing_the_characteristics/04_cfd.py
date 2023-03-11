#! /usr/bin/env python
# -*- coding:utf-8 -*-

################################################################
#
# Class 04: Solve Burgers' equation numerically and plot the characteristic curves
# Book: "The minimum you need to know to write your own CFD solver"
# Author: Pedro Stefanin Volpiani
# Online course: https://youtu.be/q-fuLz0e58Q
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
# Define parameters
#===============================================================
Nx = 201;                         # Number of grid points
xmax = 4.;                        # Domain limit to the right
xmin = -2.;                       # Domain limit to the left
dx = (xmax-xmin)/(Nx-1)           # Mesh size
x = np.arange(xmin,xmax,dx)       # Discretized mesh

t = 0.                            # Initial time
t_end = 2.                        # Final time
dt = 0.01                         # Time step
Nt = int(t_end/dt)                # Number of iterations
tt = np.linspace(0.,t_end,Nt+1)   # Time vector

# Choose initial solution !
case=4
if (case==1):
  U = np.exp( -0.5 * (x/0.4)**2 )
elif (case==2):
  U = (x<-1.)*(0)+(x<=0)*(x>-1.)*(1+x)+(x<1.)*(x>0.)*(1-x)+(x>1.)*(0)
elif (case==3):
  U = (x<=0)*(0)+(x<1.)*(x>0.)*(x)+(x>1.)*(1)
elif (case==4):
  U = (x<=0)*(1)+(x<1.)*(x>0.)*(1-x)+(x>=1.)*(0)
elif (case==5):
  U = (x<=0)*(0)+(x<1.)*(x>0.)*(1)+(x>=1.)*(0)
Uex0 = U.copy()                   # Initial solution

vec = np.zeros((Nt+1,Nx-1))       # u^n_i
n=0                               # counter

#===============================================================
# Solve equation using a conservative scheme
#===============================================================
while (t <= t_end):
  
  n = n+1
  sigma = dt/dx
  t = t+dt
  
  Un = U.copy()
  Um = np.roll(Un,1)
  Up = np.roll(Un,-1)
  Ap = 0.5 * (Up + Un)
  Am = 0.5 * (Un + Um)
  
  U = Un - 0.5 * sigma * (Am + abs(Am)) * (Un-Um) - 0.5 * sigma * (Ap - abs(Ap)) * (Up - Un)
  U[0] = Uex0[0]
  U[-1] = Uex0[-1]
  vec[n,:] = U

  #===============================================================
  # Plot solution
  #===============================================================
  if (n==1): fig, ax = plt.subplots(2)
  plt.clf()
  
  plt.subplot(2,1,1)
  plt.title('t='+str(round(dt*(n+1),3)),fontsize=16)
  plt.plot(x,U,'k')
  plt.xlabel('x',fontsize=16)
  plt.ylabel('u',fontsize=16)
  plt.axis([xmin, xmax, 0, 1.5])
  
  plt.subplot(2,1,2)
  plt.contour(x,tt,vec,np.linspace(0.1,1.,10),colors='k')
  plt.xlabel('x',fontsize=16)
  plt.ylabel('t',fontsize=16)
  plt.axis([xmin, xmax, 0, 1.5])
  
  plt.tight_layout()
  plt.draw(); plt.pause(0.001)

plt.show()
#fig.savefig("fig.pdf", dpi=300)


