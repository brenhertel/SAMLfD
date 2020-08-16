import numpy as np
import h5py
import matplotlib.pyplot as plt
import math
import time
from matplotlib.colors import LogNorm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy.interpolate import interp2d
import gradient_plotting
from sklearn.neighbors import KDTree
import os
from scipy.interpolate import RegularGridInterpolator
import seaborn as sns; sns.set()
from sklearn.svm import SVC
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import copy

DEBUG = True
COLORS = ['r', 'g', 'b', 'c', 'm', 'y']

#conventions used:
#i for x value indexing
#j for y value indexing
#k for z value indexing
#n for point indexing
#m for dimension indexing
#r for representation indexing
#p for grid_size indexing

def get_traj_dist(traj):
    dist = 0.
    for n in range(len(traj) - 2):
        dist = dist + (sum((traj[n + 1] - traj[n])**2))**0.5
    if (DEBUG):
        print('Traj total dist: %f' % (dist))
    return dist

#Meta-Lerning from Demonstration (MLfD) class
class metalfd(object):

  def __init__(self):
    self.org_traj = []
    self.n_pts = 0
    self.n_dims = 0
    self.algs = []
    self.alg_names = []
    self.n_algs = 0
    self.metrics = []
    self.n_metrics = 0
    self.is_dissim = False
    self.grid_vals = []
    self.deform_index = 0
    
  def add_traj(self, traj):
    #should by n x m array, where n is number of points and m is dimensions
    self.org_traj = traj
    (self.n_pts, self.n_dims) = np.shape(self.org_traj)
    
  def add_representation(self, alg, name=''):
    self.algs.append(alg)
    self.alg_names.append(name)
    self.n_algs = self.n_algs + 1
    
  def add_metric(self, metric, is_dissim=False):
    self.metrics.append(metric)
    self.n_metrics = 1
    self.is_dissim = is_dissim
    
  def get_demo_dist(self):
    return get_traj_dist(self.org_traj)
    
  def create_grid(self, given_grid_size=9, dists=None, index=0, plot=False):
  
    if (self.n_pts < 2) or (self.n_dims < 1):
        print('WARNING: No trajectories given')
        
    self.grid_size = given_grid_size
    self.deform_index = index
    
    if dists == None:
        K = 8.0
        dists = np.ones(self.n_dims) * (self.get_demo_dist() / K)
        
    for m in range(self.n_dims):
        center = self.org_traj[self.deform_index][m]
        self.grid_vals.append(np.linspace(center - dists[m], center + dists[m], self.grid_size))
        if (DEBUG):
            print('Grid values for dimension %d' % (m))
            print(self.grid_vals[m])
        
    
    self.deform_points = np.empty(self.n_dims)
    for gd_pt in range(self.grid_size**self.n_dims):
        index = ()
        
        new_point = []
        
        for m in reversed(range(self.n_dims)):
            this_index = int(gd_pt % self.grid_size)
            gd_pt = gd_pt - this_index
            gd_pt = gd_pt / self.grid_size
            index = (this_index,) + index
            new_point.append(self.grid_vals[m][this_index])
            
        if (DEBUG):
            print('Indexing for point %d: ' % (gd_pt))
            print(index)
            print('Point at this index: ')
            print(new_point)
            
        self.deform_points = np.vstack((self.deform_points, np.array(new_point)))
    
    self.deform_points = np.delete(self.deform_points, 0, 0)
    
    if (DEBUG):
        print('Deform Points:')
        print(self.deform_points)
        
    if (plot):
        fig = plt.figure()
        if (self.n_dims == 1):
            for gd_pt in range(self.grid_size**self.n_dims):
                plt.plot(self.deform_points[gd_pt][0], 'k.')
        elif (self.n_dims == 2):
            for gd_pt in range(self.grid_size**self.n_dims):
                plt.plot(self.deform_points[gd_pt][0], self.deform_points[gd_pt][1], 'k.')
        elif (self.n_dims == 3):
            ax = fig.add_subplot(111, projection='3d')
            for gd_pt in range(self.grid_size**self.n_dims):
                ax.plot(self.deform_points[gd_pt][0], self.deform_points[gd_pt][1], self.deform_points[gd_pt][2], 'k.')
        else:
            print('Unable to plot with current dimensions!')
        plt.show()
        plt.close('all')
        
  def deform_traj(self, plot=False):
    if (self.n_pts < 2) or (self.n_dims < 1):
        print('WARNING: No trajectories given')
    if (self.n_algs < 2):
        print('WARNING: Not enough representations given!')
  
    self.deform_trajs = [[np.zeros(np.shape(self.org_traj)) for gd_pt in range(self.grid_size**self.n_dims)] for m in range(self.n_dims)]
    
    if (DEBUG):
        print('Deform Trajectory Base:')
        print(self.deform_trajs)
            
    for gd_pt in range(self.grid_size**self.n_dims):
        for n in range(self.n_algs):
            self.deform_trajs[n][gd_pt] = self.algs[n](self.org_traj, self.deform_points[gd_pt], self.deform_index)
        
            if (DEBUG):
                print('Deform Trajectory at index %d:' % (gd_pt))
                print(self.deform_trajs[n][gd_pt])
        
    if (plot):
        fig = plt.figure()
        if (self.n_dims == 1):
            for gd_pt in range(self.grid_size**self.n_dims):
                plt.subplot(self.n_dims, self.grid_size, gd_pt + 1)
                for n in range(self.n_algs):
                    plt.plot(self.deform_trajs[n][gd_pt], COLORS[n])
                    plt.plot(self.org_traj, 'k')
        elif (self.n_dims == 2):
            for gd_pt in range(self.grid_size**self.n_dims):
                plt.subplot(self.grid_size, self.grid_size, gd_pt + 1)
                for n in range(self.n_algs):
                    if (DEBUG):
                        print('x')
                        print(self.deform_trajs[0][0][0][:, 0])
                        print('y')
                        print(self.deform_trajs[0][0][0][:, 1])
                    plt.plot(self.deform_trajs[n][gd_pt][0][:, 0], self.deform_trajs[n][gd_pt][0][:, 1], COLORS[n])
                if (DEBUG):
                    print('x')
                    print(self.org_traj[:, 0])
                    print('y')
                    print(self.org_traj[:, 1])
                plt.plot(self.org_traj[:, 0], self.org_traj[:, 1], 'k')
        else:
            print('Unable to plot with current dimensions!')
        plt.show()
        plt.close('all')
        
        
        
        
        
        
        
        
        
        
def main():
        x = np.linspace(1, 10)
        y = np.linspace(1, 10)**2
        z = np.linspace(1, 10)**3
        
        my_mlfd = metalfd()
        #my_mlfd.add_traj(np.transpose(np.vstack((x, y, z))))
        my_mlfd.add_traj(np.transpose(np.vstack((x, y))))
        import lte
        my_mlfd.add_representation(lte.LTE_ND, 'LTE')
        my_mlfd.create_grid(given_grid_size=3, plot=False)
        my_mlfd.deform_traj(plot=True)
        
if __name__ == '__main__':
  main()    
        