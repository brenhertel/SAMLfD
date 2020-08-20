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
#d for dimension indexing
#r for representation indexing
#p for grid_size indexing

#map function to map values from one min/max to another min/max
#arguments
#x: value to be mapped
#in_min: minimum value of the input
#in_max: maximum value of the input
#out_min: minimum value of the output
#out_max: maximum value of the output
#returns the input mapped to the range of the output
def my_map(x, in_min, in_max, out_min, out_max): #arduino's map function
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
    
#function to downsample a 1 dimensional trajectory to n points
#arguments
#traj: nxd vector, where n is number of points and d is number of dims
#n (optional): the number of points in the downsampled trajectory. Default is 100.
#returns the trajectory downsampled to n points
def downsample_traj(traj, n=100):
    n_pts, n_dims = np.shape(traj)
    npts = np.linspace(0, n_pts - 1, n)
    out = np.zeros((n, n_dims))
    for i in range(n):
        out[i][:] = traj[int(npts[i])][:]
    return out
    
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
    #self.metrics = [] #possibly restore multi-metric functionality
    self.metric = []
    self.n_metrics = 0
    self.is_dissim = False
    self.grid_vals = []
    self.deform_index = 0
    self.constrain_init = True
    self.constrain_end = True
    self.deform_trajs = []
    self.strongest_sims = []
    
  def add_traj(self, traj):
    #should by n x m array, where n is number of points and m is dimensions
    self.org_traj = traj
    (self.n_pts, self.n_dims) = np.shape(self.org_traj)
    
  def add_representation(self, alg, name=''):
    self.algs.append(alg)
    self.alg_names.append(name)
    self.n_algs = self.n_algs + 1
    
  def add_metric(self, metric, is_dissim=False):
    #self.metrics.append(metric)
    self.metric = metric
    self.n_metrics = 1
    self.is_dissim = is_dissim
    
  def get_demo_dist(self):
    return get_traj_dist(self.org_traj)
    
  def create_grid(self, given_grid_size=9, dists=None, index=0, plot=False):
    #check enpoint constraints
    if (index == 0):
        self.constrain_init = False
    if (index == -1) or (index == self.n_pts - 1):
        self.constrain_end = False
        
    if (self.n_pts < 2) or (self.n_dims < 1):
        print('WARNING: No trajectories given')
        
    self.grid_size = given_grid_size
    self.deform_index = index
    
    if dists == None:
        K = 8.0
        dists = np.ones(self.n_dims) * (self.get_demo_dist() / K)
        
    for d in range(self.n_dims):
        center = self.org_traj[self.deform_index][d]
        self.grid_vals.append(np.linspace(center - dists[d], center + dists[d], self.grid_size))
        if (DEBUG):
            print('Grid values for dimension %d' % (d))
            print(self.grid_vals[d])
        
    
    self.deform_points = np.empty(self.n_dims)
    for gd_pt in range(self.grid_size**self.n_dims):
        index = ()
        
        new_point = []
        
        for d in reversed(range(self.n_dims)):
            this_index = int(gd_pt % self.grid_size)
            gd_pt = gd_pt - this_index
            gd_pt = gd_pt / self.grid_size
            index = (this_index,) + index
            new_point.append(self.grid_vals[d][this_index])
            
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
  
    self.deform_trajs = [[np.zeros(np.shape(self.org_traj)) for m in range(self.n_dims)] for gd_pt in range(self.grid_size**self.n_dims)]
    
    if (DEBUG):
        print('Deform Trajectory Base:')
        print(self.deform_trajs)
            
    for gd_pt in range(self.grid_size**self.n_dims):
        #set up constraints
        constraints = [self.deform_points[gd_pt]]
        constrain_indexes = [self.deform_index]
        if (self.constrain_init):
            constraints = np.vstack((constraints, self.org_traj[0]))
            constrain_indexes = np.vstack((constrain_indexes, [0]))
        if (self.constrain_end):
            constraints = np.vstack((constraints, self.org_traj[self.n_pts - 1]))
            constrain_indexes = np.vstack((constrain_indexes, [self.n_pts - 1]))
            
        for r in range(self.n_algs):
            self.deform_trajs[gd_pt][r] = self.algs[r](self.org_traj, constraints, constrain_indexes)
        
            if (DEBUG):
                print('Deform Trajectory at index %d:' % (gd_pt))
                print(self.deform_trajs[gd_pt][r])
        
    if (plot):
        fig = plt.figure()
        if (self.n_dims == 1):
            for gd_pt in range(self.grid_size**self.n_dims):
                plt.subplot(self.n_dims, self.grid_size, gd_pt + 1)
                for r in range(self.n_algs):
                    plt.plot(self.deform_trajs[gd_pt][r], COLORS[r])
                    plt.plot(self.org_traj, 'k')
        elif (self.n_dims == 2):
            for gd_pt in range(self.grid_size**self.n_dims):
                plt.subplot(self.grid_size, self.grid_size, gd_pt + 1)
                for r in range(self.n_algs):
                    if (DEBUG):
                        print('x')
                        print(self.deform_trajs[gd_pt][r][:, 0])
                        print('y')
                        print(self.deform_trajs[gd_pt][r][:, 1])
                    plt.plot(self.deform_trajs[gd_pt][r][:, 0], self.deform_trajs[gd_pt][r][:, 1], COLORS[r])
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
        
  def calc_similarities(self, downsample=True):
    if (self.deform_trajs == []):
        print('WARNING: No deformed trajectories found, quitting')
        exit()
    if (self.n_metrics < 1):
        print('WARNING: No metric given!')
  
    self.sim_vals = np.zeros((self.grid_size**self.n_dims, self.n_algs))
    
    if (DEBUG):
        print('Similarity Value Base:')
        print(self.sim_vals)
            
    for gd_pt in range(self.grid_size**self.n_dims):
        for r in range(self.n_algs):
            if (downsample):
                self.sim_vals[gd_pt][r] = self.metric(downsample_traj(self.org_traj), downsample_traj(self.deform_trajs[gd_pt][r]))
            else:
                self.sim_vals[gd_pt][r] = self.metric(self.org_traj, self.deform_trajs[gd_pt][r])
        
    min_sim = np.amin(self.sim_vals)
    max_sim = np.amax(self.sim_vals)
    
    for gd_pt in range(self.grid_size**self.n_dims):
        for r in range(self.n_algs):
            if (self.is_dissim):
                self.sim_vals[gd_pt][r] = my_map(self.sim_vals[gd_pt][r], min_sim, max_sim, 1, 0) 
            else:
                self.sim_vals[gd_pt][r] = my_map(self.sim_vals[gd_pt][r], min_sim, max_sim, 0, 1) 
    self.get_strongest_sims()
        
  def plot_heatmap(self, mode='save', filepath=''):
    map_size = ()
    for d in range(self.n_dims):
        map_size = map_size + (self.grid_size, )
   
    for r in range(self.n_algs):
        name = self.alg_names[r] + ' Heatmap'
        if (DEBUG):
            print(name)
        
        A = np.zeros(map_size)
        
        for gd_pt in range(self.grid_size**self.n_dims):
            gd_pt_org = gd_pt
            index = ()
            for d in range(self.n_dims):
                this_index = int(gd_pt % self.grid_size)
                gd_pt = gd_pt - this_index
                gd_pt = gd_pt / self.grid_size
                index = (this_index,) + index
            if (DEBUG):
                print(self.sim_vals[gd_pt_org][r])
            A[index] = self.sim_vals[gd_pt_org][r]
        if (DEBUG):
            print(A)
        ax = sns.heatmap(np.transpose(A), annot=False)
        plt.xticks([])
        plt.yticks([])
        if (mode == 'save'):
            plt.savefig(filepath + name + '.png')
        else:            
            plt.show() 
        plt.close('all')
        
        
  def save_to_h5(self, filename='unspecified.h5'):
    fp = h5py.File(filename, 'w')
    dset_name = 'mlfd'
    #save grid_size
    fp.create_dataset(dset_name + '/grid_sz', data=self.grid_size)
    #save grid_vals
    for d in range(self.n_dims):
        fp.create_dataset(dset_name + '/grid_vals/' + str(d), data=self.grid_vals[d])
    #save deform points
    fp.create_dataset(dset_name + '/deform_points', data=self.deform_points)
    #save deform trajectories
    for gd_pt in range(self.grid_size**self.n_dims):
        for r in range(self.n_algs):
            fp.create_dataset(dset_name + '/deform_trajs/' + str(gd_pt) + '/' + str(r), data = self.deform_trajs[gd_pt][r])
    #save similarity values
    fp.create_dataset(dset_name + '/similarity_values', data=self.sim_vals)
    fp.close()
    
  def load_from_h5(self, filename):
    fp = h5py.File(filename, 'r')
    dset_name = 'mlfd'
    dset = fp.get(dset_name)
    #get grid_size
    self.grid_size = np.array(dset.get('grid_sz'))
    if (DEBUG):
        print(self.grid_size)
    #get grid_vals
    grid_data = dset.get('grid_vals')
    self.grid_vals = [None] * self.n_dims
    for d in range(self.n_dims):
        self.grid_vals[d] = np.array(grid_data.get(str(d)))
    #get deform points
    self.deform_points = np.array(dset.get('deform_points'))
    #get deform trajectories
    self.deform_trajs = [[np.zeros(np.shape(self.org_traj)) for m in range(self.n_dims)] for gd_pt in range(self.grid_size**self.n_dims)]
    deform_data = dset.get('deform_trajs')
    for gd_pt in range(self.grid_size**self.n_dims):
        gd_pt_data = deform_data.get(str(gd_pt))
        for r in range(self.n_algs):
            self.deform_trajs[gd_pt][r] = np.array(gd_pt_data.get(str(r)))
    #get similarity values
    self.sim_vals = np.array(dset.get('similarity_values'))
    fp.close()
        
  def get_strongest_sims(self, treshold = 0.0):
    self.strongest_sims = np.zeros(self.grid_size**self.n_dims)
    for gd_pt in range(self.grid_size**self.n_dims):
        max_val = self.sim_vals[gd_pt][0]
        second_max = max_val
        max_ind = 0
        for r in range(self.n_algs):
            if (self.sim_vals[gd_pt][r] > max_val):
                if (max_val > second_max):
                    second_max = max_val
                max_val = self.sim_vals[gd_pt][r]
                max_ind = r
            if (self.sim_vals[gd_pt][r] < max_val) and (self.sim_vals[gd_pt][r] > second_max):
                second_max = self.sim_vals[gd_pt][r]
        if (max_val - second_max < treshold):
            self.strongest_sims[gd_pt] = -1
        else:
            self.strongest_sims[gd_pt] = max_ind
        
  def plot_strongest_sims(self):
    map_size = ()
    for d in range(self.n_dims):
        map_size = map_size + (self.grid_size, )
    
    for gd_pt in range(self.grid_size**self.n_dims):
        gd_pt_org = gd_pt
        index = ()
        for d in range(self.n_dims):
            this_index = int(gd_pt % self.grid_size)
            gd_pt = gd_pt - this_index
            gd_pt = gd_pt / self.grid_size
            index = (this_index,) + index
        
        
def main():
        x = np.linspace(1, 10)
        y = np.linspace(1, 10)**2
        z = np.linspace(1, 10)**3
        
        my_mlfd = metalfd()
        #my_mlfd.add_traj(np.transpose(np.vstack((x, y, z))))
        my_mlfd.add_traj(np.transpose(np.vstack((x, y))))
        import ja
        my_mlfd.add_representation(ja.perform_ja_general, 'JA')
        import lte
        my_mlfd.add_representation(lte.LTE_ND_any_constraints, 'LTE')
        import similaritymeasures
        my_mlfd.add_metric(similaritymeasures.frechet_dist, is_dissim=True)
        #my_mlfd.create_grid(given_grid_size=3, plot=False)
        #my_mlfd.deform_traj(plot=True)
        #my_mlfd.calc_similarities()
        #my_mlfd.save_to_h5(filename='test.h5')
        my_mlfd.load_from_h5(filename='test.h5')
        my_mlfd.plot_heatmap(mode='show')
        
if __name__ == '__main__':
  main()    
        