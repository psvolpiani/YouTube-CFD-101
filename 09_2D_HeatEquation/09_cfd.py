#! /usr/bin/env python
# -*- coding:utf-8 -*-

################################################################
#
# Class 10: Solve the 2D heat equation using a 2-order central-difference scheme
# Book: "The minimum you need to know to write your own CFD solver"
# Author: Pedro Stefanin Volpiani
# Online course: https://youtu.be/GPalSj1Bpqc
#
################################################################

import numpy as np
import matplotlib.pyplot as plt

#===============================================================
# Parameters
#===============================================================
Lx = 1.; Ly = 1.          # Box lenghts
nx = 31; ny = 31          # Number of points
dx = Lx/(nx-1)            # Step size
dy = Ly/(ny-1)            # Step size
x = np.linspace(0,Lx,nx)  # Mesh in x
y = np.linspace(0,Ly,ny)  # Mesh in y
alpha = 1.0               # Diffusion coefficient
niter = 1000              # Number of iterations
dt    = (dx**2)/(4*alpha) # time step respecting the stability condition
t_end = 10                # Final time
ncolour = 20              # Number of bands
colourMap=plt.cm.jet      # Colour map

#===============================================================
# Boundary conditions (fixed temperature)
#===============================================================
u_top = 100.0
u_left = 90.0
u_bottom = 90.0
u_right = 90.0

#===============================================================
# Compute the exact solution
#===============================================================
def compute_u_exact(u):
  sum=0;
  for i in range (0,nx):
    for j in range (0,ny):
      for n in range (1,200):
        sum = ((-1)**(n+1)+1.)/n *np.sin(n*np.pi*x[i]/Lx)* (np.sinh(n*np.pi*y[j]/Lx)/np.sinh(n*np.pi*Ly/Lx)) + sum
      u[i,j] = 2./np.pi*sum*(u_top-u_bottom)+u_bottom
      sum=0
  return u

#===============================================================
# Compute Laplace operator
#===============================================================
def compute_laplace2d(u):
  dudt = np.zeros((nx,ny))
  for i in range(1, nx-1):
    for j in range(1, ny-1):
      dudt[i,j] = alpha*( (u[i+1,j]-2*u[i,j]+u[i-1,j])/dx**2 + (u[i,j+1]-2*u[i,j]+u[i,j-1])/dy**2)
  return dudt

#===============================================================
# Set the boundary conditions
#===============================================================
def apply_BCs(u):
  u[:,-1] = u_top
  u[0, :] = u_left
  u[:, 0] = u_bottom
  u[-1,:] = u_right
  return u
  
#===============================================================
# Main program
#===============================================================
uex = np.zeros((nx,ny))
uex = compute_u_exact(uex)

# Initial condition
u = np.ones((nx,ny))*90
u = apply_BCs(u)

t = 0.; n = 0;
# Temporal loop
while t < t_end:

  dudt = compute_laplace2d(u)
  u = u + dudt*dt
  u = apply_BCs(u)
    
  # Update time and iteration counter
  t=t+dt; n=n+1;
  
  # Criterion
  if ( sum(sum(dudt)) < 1.e-3 ): break
  
fig=plt.figure(0,figsize=(5.5,4))
plt.xlabel("x"); plt.ylabel("y");
plt.contourf(x, y, u.transpose(), ncolour, cmap=colourMap, vmin=u_bottom, vmax=u_top)
plt.colorbar()
plt.contour(x,y,uex.transpose(), ncolour, colors='white', vmin=u_bottom, vmax=u_top)
plt.axis('tight')
plt.show()
