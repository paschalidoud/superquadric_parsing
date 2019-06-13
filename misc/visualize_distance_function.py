#!/usr/bin/env python

import os
import sys

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import numpy as np

def C(points, shapes, epsilons):
    assert points.shape[0] == 3
    assert shapes.shape[0] == 3
    assert epsilons.shape[0] == 2
    
    a1 = shapes[0]
    a2 = shapes[1]
    a3 = shapes[2]
    e1 = epsilons[0]
    e2 = epsilons[1]
    
    # zeros = points == 0
    # points[zeros] = points[zeros] + 1e-6
    
    F = ((points[0, :] / a1)**2.0)**(1.0/e2)
    F += ((points[1, :] / a2)**2.0)**(1.0/e2)
    F = F**(e2/e1)
    F += ((points[2, :] / a3)**2.0)**(1.0/e1)
    return F**e1 - 1.0



def get_C(a1, a2, a3, e1, e2, val, plane="z"):
    if val > a3 and plane == "z":
         return
    elif val > a2 and plane == "y":
        return
    elif val > a3 and plane == "z":
        return
    x = np.linspace(-0.5, 0.5, 100)
    y = np.linspace(-0.5, 0.5, 100)
    xv, yv = np.meshgrid(x, y)
    if plane == "z":
        points  = np.stack([
           xv.ravel(),
           yv.ravel(),
           np.ones_like(xv.ravel())*val
       ])
    elif plane == "y":
        points = np.stack([
           xv.ravel(),
           np.ones_like(xv.ravel())*val,
           yv.ravel()
       ])
    elif plane == "x":
        points = np.stack([
           np.ones_like(xv.ravel())*val,
           xv.ravel(),                                        
           yv.ravel()                                                          
        ])
    z = C(
       points,
       np.array([[a1, a2, a3]]).ravel(),
       np.array([[e1, e2]]).ravel()
    )
    return xv, yv, z


def plot_C(a1, a2, a3, e1, e2, val, i):
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12,12))
    xv, yv, z = get_C(a1, a2, a3, e1, e2, val[0], "z")
    cs = ax1.contourf(xv, yv, z.reshape(100, 100), cmap=cm.PuBu_r, vmin=z.min(), vmax=z.max())
    cbar = fig.colorbar(cs, ax=ax1)
    ax1.set_title("Z-plane at %.5f a1: %.2f, a2: %.2f, a3: %.2f, e1: %.2f, e2: %.2f" %(
        val[0], a1, a2, a3, e1, e2
    ))
    xv, yv, z = get_C(a1, a2, a3, e1, e2, val[1], "y")
    cs = ax2.contourf(xv, yv, z.reshape(100, 100), cmap=cm.PuBu_r, vmin=z.min(), vmax=z.max())
    ax2.set_title("Y-plane at %.5f a1: %.2f, a2: %.2f, a3: %.2f, e1: %.2f, e2: %.2f" %(
        val[1], a1, a2, a3, e1, e2
    ))
    cbar = fig.colorbar(cs, ax=ax2)
    xv, yv, z = get_C(a1, a2, a3, e1, e2, val[1], "x")
    cs = ax3.contourf(xv, yv, z.reshape(100,100), cmap=cm.PuBu_r, vmin=z.min(), vmax=z.max())
    ax3.set_title("X-plane at %.5f a1: %.2f, a2: %.2f, a3: %.2f, e1: %.2f, e2: %.2f" %(
        val[2], a1, a2, a3, e1, e2
    ))
    cbar = fig.colorbar(cs, ax=ax3)
    plt.savefig("/tmp/C_%03d.png" %(i))
    # plt.show()
    plt.close()


if __name__ == "__main__":
    N = 100
    planes = [0.004, 0.006, 0.003]
    a1s = np.random.random((N,))*0.5 + 1e-2
    a2s = np.random.random((N,))*0.5 + 1e-2
    a3s = np.random.random((N,))*0.5 + 1e-2
    e1s = np.random.random((N,))*1.6 + 0.2
    e2s = np.random.random((N,))*1.6 + 0.2

    for i, a1, a2, a3, e1, e2 in zip(range(N), a1s, a2s, a3s, e1s, e2s):
        print i, a1, a2, a3, e1, e2
        plot_C(a1, a2, a3, e1, e2, planes, i)
