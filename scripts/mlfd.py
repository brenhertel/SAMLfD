import numpy as np
import h5py
import matplotlib.pyplot as plt
import math
import time
from matplotlib.colors import LogNorm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import os
from scipy.interpolate import RegularGridInterpolator
import seaborn as sns; sns.set()
from sklearn.svm import SVC
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LinearSegmentedColormap
import copy

DEBUG = False
COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] #black is for underflow with queries that don't meet similarity. Not used for representations.

#conventions used:
#i, t for x value indexing
#j, u for y value indexing
#k, v for z value indexing
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

#function to get the total distance of a n x d trajectory
#arguments
#traj: nxd vector, where n is number of points and d is number of dims
#returns the total distance of traj, calculated using euclidean distance  
def get_traj_dist(traj):
    dist = 0.
    for n in range(len(traj) - 2):
        dist = dist + (sum((traj[n + 1] - traj[n])**2))**0.5
    if (DEBUG):
        print('Traj total dist: %f' % (dist))
    return dist

def convert_num_to_rgb(input):
    out_triplet = np.zeros(3)
    if input == 0:
        #red
        out_triplet[0] = 1
    elif input == 1:
        #green
        out_triplet[1] = 1
    elif input == 2:
        #blue
        out_triplet[2] = 1
    elif input == 3:
        #yellow
        out_triplet[0] = 1
        out_triplet[1] = 1
    elif input == 4:
        #magenta
        out_triplet[0] = 1
        out_triplet[2] = 1
    elif input == 5:
        #cyan
        out_triplet[1] = 1
        out_triplet[2] = 1
    elif (input > 5):
        print('Too many algorithms to represent color')
    else:
        #black
        print()
    return out_triplet

#Meta-Lerning from Demonstration (MLfD) class
class metalfd(object):

  #initialize variables
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

#########
# UTILITY
#########

  #function to get the total distance of the trajectory stored in Meta-Lfd
  #no arguments
  #returns the total distance of the trajectory
  def get_demo_dist(self):
    return get_traj_dist(self.org_traj)
    
  #function to convert from gd_pt values to index values. Useful for plotting.
  #arguments
  #gd_pt: a scalar representing the grid point index
  #return the index (as a tuple) that the grid point maps to
  def convert_gd_pt_to_index(self, gd_pt):
    index = ()
    for d in range(self.n_dims):
        this_index = int(gd_pt % self.grid_size)
        gd_pt = gd_pt - this_index
        gd_pt = gd_pt / self.grid_size
        index = (this_index,) + index
    return index
  
  #plots the outline of a rectangular prism that encompasses the similarity region on an already created 3D plot.
  #arguments
  #ax: handle to the 3D axes object
  #no returns
  def get_cube_outline(self, ax):
    min_x = self.grid_vals[0][0]
    max_x = self.grid_vals[0][self.grid_size - 1]
    min_y = self.grid_vals[1][0]
    max_y = self.grid_vals[1][self.grid_size - 1]
    min_z = self.grid_vals[2][0]
    max_z = self.grid_vals[2][self.grid_size - 1]
    
    #squares in x-y plane
    xv = [min_x, min_x, max_x, max_x]
    yv = [min_y, max_y, max_y, min_y]
    zv = [min_z, min_z, min_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_alpha(0.1)
    poly.set_facecolor('w')
    ax.add_collection3d(poly)
    xv = [min_x, min_x, max_x, max_x]
    yv = [min_y, max_y, max_y, min_y]
    zv = [max_z, max_z, max_z, max_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_alpha(0.1)
    poly.set_facecolor('w')
    ax.add_collection3d(poly)
    
    #squares in x-z plane
    xv = [min_x, min_x, max_x, max_x]
    yv = [min_y, min_y, min_y, min_y]
    zv = [min_z, max_z, max_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_alpha(0.1)
    poly.set_facecolor('w')
    ax.add_collection3d(poly)
    xv = [min_x, min_x, max_x, max_x]
    yv = [max_y, max_y, max_y, max_y]
    zv = [min_z, max_z, max_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_alpha(0.1)
    poly.set_facecolor('w')
    ax.add_collection3d(poly)
    
    #squares in y-z plane
    xv = [min_x, min_x, min_x, min_x]
    yv = [min_y, min_y, max_y, max_y]
    zv = [min_z, max_z, max_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_alpha(0.1)
    poly.set_facecolor('w')
    ax.add_collection3d(poly)
    xv = [max_x, max_x, max_x, max_x]
    yv = [min_y, min_y, max_y, max_y]
    zv = [min_z, max_z, max_z, min_z]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_alpha(0.1)
    poly.set_facecolor('w')
    ax.add_collection3d(poly)

###############
# STEP 1: SETUP
###############    
    
  #function that sets gives the input demonstration for Meta-LfD
  #arguments
  #traj: nxd vector, where n is number of points and d is number of dims
  #no return
  def add_traj(self, traj):
    #should by n x m array, where n is number of points and m is dimensions
    self.org_traj = traj
    (self.n_pts, self.n_dims) = np.shape(self.org_traj)
    if (DEBUG):
        print(self.org_traj)
        print((self.n_pts, self.n_dims))
    
  #function that gives a LfD representation to Meta-LfD
  #arguments
  #alg: function pointer to LfD representation. Inputs to function should be formatted f(org_traj, constraints, index) and the only return of the function should be the deformed trajectory, formatted in the same shape as the original trajectory.
  #name: name of the representation. Used for plotting.
  #no return 
  def add_representation(self, alg, name=''):
    self.algs.append(alg)
    self.alg_names.append(name)
    self.n_algs = self.n_algs + 1
   
  #function that gives a similarity metric to Meta-LfD
  #arguments
  #metric: function pointer to similarity metric. Inputs to function should be formatted f(org_traj, compare_traj) and the only return of the function should be a scalar value representing the similarity.
  #is_dissim: boolean that describes if the metric measures similarity or dissimilarity. Similarity means higher values are more similar, dissimilarity means higher values are less similar.
  #no return   
  def add_metric(self, metric, is_dissim=False):
    #self.metrics.append(metric)
    self.metric = metric
    self.n_metrics = 1
    self.is_dissim = is_dissim

  #function that plots the original trajectory
  #arguments:
  #mode: a string, either 'save' or anything else. If 'save' the plot gets saved to the specified filepath. Otherwise it is shown to the user.
  #filepath: filepath to directory where plot is saved to
  #no returns
  def plot_org_traj(self, mode='save', filepath=''):
    fig = plt.figure()
    name = 'Original Trajectory'
    if (self.n_dims == 1):
        plt.plot(self.org_traj, 'k')
    elif (self.n_dims == 2):
        plt.plot(self.org_traj[:, 0], self.org_traj[:, 1], 'k')
    elif (self.n_dims == 3):
            ax = fig.add_subplot(111, projection='3d')
            plt.plot(self.org_traj[:, 0], self.org_traj[:, 1], self.org_traj[:, 2], 'k')
    else:
        print('Incorrect dimensionality to be able to plot')
    if (mode == 'save'):
        plt.savefig(filepath + name + '.png')
    else:            
        plt.show()
    plt.close('all')

##################
# STEP 2: MESHGRID
################## 
    
  #function to create the Meta-LfD grid which is then used to deform the demonstration
  #arguments
  #given_grid_size (optional): the side length of the ragular structure which composes the deformation area (the side length of the square for 2D or cube for 3D). 9 was found to be a good default value experimentally.
  #dists (optional): an array of d values of the side length of the deformation area, where d is the dimensionality of the trajectory. If none are given, the default value of 1/8 the length of the trajectory is used (this value was experimentally determined to be an adequate estimation of the size of the workspace).
  #index (optional): the index to be deformed. If none is chosen, the default of initial point is used. Note: if using endpoint, please use (len(traj) - 1) as index instead of -1.
  #plot (optional): if true, plots each grid point. Note: only works for up to 3D.
  #no return
  def create_grid(self, given_grid_size=9, dists=None, index=0, plot=False):
  
    #check enpoint constraints
    #by default we constrain the endpoints, and we can't deforma and constrain an endpoint at the same time
    if (index == 0):
        self.constrain_init = False
    if (index == -1) or (index == self.n_pts - 1):
        self.constrain_end = False
        
    if (self.n_pts < 2) or (self.n_dims < 1):
        print('WARNING: No trajectories given')
        
    self.grid_size = given_grid_size
    self.deform_index = index
    
    #get default dist if none chosen
    if dists == None:
        K = 8.0
        dists = np.ones(self.n_dims) * (self.get_demo_dist() / K)
        
    #create the grid of deformations. Using linspace creates linearly spaced points for each dimension
    for d in range(self.n_dims):
        center = self.org_traj[self.deform_index][d]
        self.grid_vals.append(np.linspace(center - dists[d], center + dists[d], self.grid_size))
        if (DEBUG):
            print('Starting position')
            print(self.org_traj[self.deform_index])
            print('Grid values for dimension %d' % (d))
            print(self.grid_vals[d])
        
    #initialize this array. This row will be deleted later
    self.deform_points = np.empty(self.n_dims)
    #go through and create an gd_pt x d array, where each gd_pt has a 1-to-1 mapping to a point in the grid. Using a 2D array was simpler to iterate through and works for ND.
    for gd_pt in range(self.grid_size**self.n_dims):
        index = ()
        
        new_point = []
        
        #Convert grid point scalar to an index value
        for d in range(self.n_dims):
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
      
    #plot if requested
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

##########################
# STEP 3: DEFORM FROM GRID
########################## 
        
  #function that deforms the demonstration using the constraints given at each grid point for each representation
  #arguments
  #plot (optional): if true, will show the deformation at each grid point. Only works for up to 2D.
  #no return
  def deform_traj(self, plot=False):
    if (self.n_pts < 2) or (self.n_dims < 1):
        print('WARNING: No trajectories given')
    if (self.n_algs < 2):
        print('WARNING: Not enough representations given!')
  
    #create a 2D list of arrays to store the deformed trajectories. (I could technically get the deformation, get the similarity, and then not record the deformed trajectory as it is no longer used in processing. I do not do this as it has been useful to see behaviors. However, if you are limited on time or space, I would suggest this method).
    self.deform_trajs = [[np.zeros(np.shape(self.org_traj)) for r in range(self.n_algs)] for gd_pt in range(self.grid_size**self.n_dims)]
    
    if (DEBUG):
        print('Deform Trajectory Base:')
        print(self.deform_trajs)
    
    #for reach grid point
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
        
        #for each algorithm
        for r in range(self.n_algs):
            #perform & record deformation
            
            if (DEBUG):
                print('Grid point: %d, Algorithm number: %d' % (gd_pt, r))
            
            self.deform_trajs[gd_pt][r] = self.algs[r](self.org_traj, constraints, constrain_indexes)
        
            if (DEBUG):
                print('Deform Trajectory at index %d:' % (gd_pt))
                print(self.deform_trajs[gd_pt][r])
    
    #plot if requested
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

###############################
# STEP 4: SIMILARITY EVALUATION
###############################

  #function that calculates the similarities of each deformation at each grid point
  #arguments
  #downsample (optional): if true, will downsample the original and reproduced trajectories to 100 points per dimension. If there are 100 or less points in the trajectory, this will be set to false.
  #no return   
  def calc_similarities(self, downsample=True):
    if (self.deform_trajs == []):
        print('WARNING: No deformed trajectories found, quitting')
        exit()
    if (self.n_metrics < 1):
        print('WARNING: No metric given!')
  
    #ensure downsample is actually downsampling
    if (self.n_pts <= 100):
        downsample = False
  
    #set up array to store similarity values
    self.sim_vals = np.zeros((self.grid_size**self.n_dims, self.n_algs))
    
    if (DEBUG):
        print('Similarity Value Base:')
        print(self.sim_vals)
    
    #for each deformation point
    for gd_pt in range(self.grid_size**self.n_dims):
        #for each representation deformation
        for r in range(self.n_algs):
            #calculate similarity
            if (downsample):
                self.sim_vals[gd_pt][r] = self.metric(downsample_traj(self.org_traj), downsample_traj(self.deform_trajs[gd_pt][r]))
            else:
                self.sim_vals[gd_pt][r] = self.metric(self.org_traj, self.deform_trajs[gd_pt][r])
    
    #find the absolute max and min of the similarity values.
    min_sim = np.amin(self.sim_vals)
    max_sim = np.amax(self.sim_vals)
    
    #map the similarity values from their current scale to a 0 to 1 scale, where 1 is greatest similarity
    for gd_pt in range(self.grid_size**self.n_dims):
        for r in range(self.n_algs):
            if (self.is_dissim):
                self.sim_vals[gd_pt][r] = my_map(self.sim_vals[gd_pt][r], min_sim, max_sim, 1, 0) 
            else:
                self.sim_vals[gd_pt][r] = my_map(self.sim_vals[gd_pt][r], min_sim, max_sim, 0, 1) 
   
  #plots a heatmap using the similarity values. Note: only works for up to 2D deformations (Functionality to create "heat cubes" will be added in future)
  #arguments:
  #mode: a string, either 'save' or anything else. If 'save' the plot gets saved to the specified filepath. Otherwise it is shown to the user.
  #filepath: filepath to directory where heatmap image is saved to
  #no returns
  def plot_heatmap(self, mode='save', filepath=''):
    if (self.n_dims < 3):
        map_size = ()
        for d in range(self.n_dims):
            map_size = map_size + (self.grid_size, )
    
        for r in range(self.n_algs):
            name = self.alg_names[r] + ' Heatmap'
            if (DEBUG):
                print(name)
            
            sim_vals_square = np.zeros(map_size)
            
            for gd_pt in range(self.grid_size**self.n_dims):
                index = self.convert_gd_pt_to_index(gd_pt)
                if (DEBUG):
                    print(self.sim_vals[gd_pt][r])
                sim_vals_square[index] = self.sim_vals[gd_pt][r]
            if (DEBUG):
                print(sim_vals_square)
            ax = sns.heatmap(np.transpose(sim_vals_square), annot=False, vmin=0, vmax=1)
            plt.xticks([])
            plt.yticks([])
            if (mode == 'save'):
                plt.savefig(filepath + name + '.png')
            else:            
                plt.show() 
            plt.close('all')
    elif (self.n_dims == 3):
        for r in range(self.n_algs):
            name = self.alg_names[r] + ' Heatcube'
            #set up coordinates
            x, y, z = np.indices((self.grid_size, self.grid_size, self.grid_size))
            #create cube
            voxels = (x >= 0) & (y >= 0) & (z >= 0)
            #create cube with colors
            colors = np.zeros(voxels.shape + (3,))
            #define colors
            for gd_pt in range(self.grid_size**self.n_dims):
                ind = self.convert_gd_pt_to_index(gd_pt)
                colors[ind + ((r % 3),)] = self.sim_vals[gd_pt][r]
            # and plot everything
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.voxels(voxels, facecolors=colors, edgecolor='k')
            if (mode == 'save'):
                plt.savefig(filepath + name + '.png')
            else:            
                plt.show() 
            plt.close('all')
    else:
        print('Incorrect dimensionality to be able to plot')
            

#####################
# STEP 4.5: SAVE/LOAD
#####################
  
  #saves the data in the current Meta-LfD framework process. Note: it is highly recommended you do this for each demonstration. Going through and getting each deform can be a lengthy process, whereas it takes fractions of a second to load it from a file. This should be done after calculating similarities but before user choice in the framework flow diagram.
  #arguments
  #filename: full filpath including filename of the .h5 file for which the information should be stored
  #no returns
  def save_to_h5(self, filename='unspecified.h5'):
    fp = h5py.File(filename, 'w')
    dset_name = 'mlfd'
    #save deform_index
    fp.create_dataset(dset_name + '/deform_index', data=self.deform_index)
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
  
  #loads data from a previously saved .h5 file.
  #filename: full filpath including filename of the .h5 file for which the information was stored
  #no returns  
  def load_from_h5(self, filename):
    fp = h5py.File(filename, 'r')
    dset_name = 'mlfd'
    dset = fp.get(dset_name)
    #get deform index
    self.deform_index = np.array(dset.get('deform_index'))
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
    self.deform_trajs = [[np.zeros(np.shape(self.org_traj)) for r in range(self.n_algs)] for gd_pt in range(self.grid_size**self.n_dims)]
    deform_data = dset.get('deform_trajs')
    for gd_pt in range(self.grid_size**self.n_dims):
        gd_pt_data = deform_data.get(str(gd_pt))
        for r in range(self.n_algs):
            self.deform_trajs[gd_pt][r] = np.array(gd_pt_data.get(str(r)))
    #get similarity values
    self.sim_vals = np.array(dset.get('similarity_values'))
    fp.close()
 
##############################
# STEP 5: INTERPRET SIMILARITY
##############################
  
  #gets the strongest similarity at each point on the grid. Used to set up classifier. Note: if 2 representations have the same similarity at a deform point, the one that was given to the framework first is chosen as the greatest.
  #arguments
  #threshold (optional): The threshold for which anything below should not be considered in creating reproductions.
  def get_strongest_sims(self, threshold=0.1):
    self.strongest_sims = np.zeros(self.grid_size**self.n_dims)
    for gd_pt in range(self.grid_size**self.n_dims):
        cur_sim_vals = self.sim_vals[gd_pt]
        if (np.amax(cur_sim_vals) >= threshold):
            self.strongest_sims[gd_pt] = np.argmax(cur_sim_vals)
        else:
            self.strongest_sims[gd_pt] = -1

  #plots the strongest similarity value at each index in its respective color. Note: This only handles up to 3 representations right now, one for each primary color. More may be added later. Additionally, only works for up to 2 dimensions
  #arguments:
  #mode: a string, either 'save' or anything else. If 'save' the plot gets saved to the specified filepath. Otherwise it is shown to the user.
  #filepath: filepath to directory where plot is saved to
  #no returns
  def plot_strongest_sims(self, mode='save', filepath=''):
    if (self.n_dims < 3):
        name = 'Strongest Similarity Representations'
        
        map_size = ()
        for d in range(self.n_dims):
            map_size = map_size + (self.grid_size, )
        
        map_size = map_size + (3,)
        
        strongest_sims_img = np.zeros(map_size)
        
        for gd_pt in range(self.grid_size**self.n_dims):
            index = self.convert_gd_pt_to_index(gd_pt)
            strongest_sims_img[index][:] = convert_num_to_rgb(self.strongest_sims[gd_pt])
            if (DEBUG):
                print('Representation with max similarity')
                print(self.strongest_sims[gd_pt])
                print('Index')
                print(index)
                print('Color')
                print(strongest_sims_img[index])
        im = plt.imshow(strongest_sims_img, vmax=1)
        plt.axis('off')
        if (mode == 'save'):
            plt.savefig(filepath + name + '.png')
        else:            
            plt.show() 
        plt.close('all')
    elif (self.n_dims == 3):
        name = 'Strongest Similarity Representations'
        #set up coordinates
        x, y, z = np.indices((self.grid_size, self.grid_size, self.grid_size))
        #create cube
        voxels = (x >= 0) & (y >= 0) & (z >= 0)
        #create cube with colors
        colors = np.zeros(voxels.shape + (3,))
        if (DEBUG):
            print('Colors shape')
            print(np.shape(colors))
        #define colors
        for gd_pt in range(self.grid_size**self.n_dims):
            ind = self.convert_gd_pt_to_index(gd_pt)
            if (DEBUG):
                print('Index:')
                print(ind)
                print('Colors at index:')
                print(colors[ind])
                print('Color change to:')
                print(convert_num_to_rgb(self.strongest_sims[gd_pt]))
            colors[ind] = convert_num_to_rgb(self.strongest_sims[gd_pt])
        # and plot everything
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(voxels, facecolors=colors, edgecolor='k')
        if (mode == 'save'):
            plt.savefig(filepath + name + '.png')
        else:            
            plt.show() 
        plt.close('all')
    else:
        print('Incorrect dimensionality to be able to plot')
        

###########################
# STEP 6: SET UP CLASSIFIER
###########################
  
  #sets up the classifier which is then used after user selects output. The function get_strongest_sims must be called before this function.
  #no arguments
  #no returns
  def set_up_classifier(self):
    self.clf = SVC()
    self.clf.fit(self.deform_points, self.strongest_sims)

##############################
# STEP 7: USER-DESIRED OUTPUTS
##############################
  
  #crafts a reproduction with greatest similarity from requested point
  #arguments
  #coord: a 2D array structured [[x1, x2, ..., xN]] which indicates the point from which the classifier should be queried as well as the constraint on the reproduced trajectory
  #plot (optional): if True, the reproduction is plotted. Note: only works in 3D or less.
  #returns the reproduced trajectory  
  def reproduce_at_point(self, coords, plot=False):
    #get algorithm with greatest similarity
    opt_alg_num = int(self.clf.predict(coords))
    
    if (DEBUG):
        print('Algorithm with greatest similarity: ' + self.alg_names[opt_alg_num])
        
    #set up constraints
    #check enpoint constraints
    #by default we constrain the endpoints, and we can't deforma and constrain an endpoint at the same time
    if (self.deform_index == 0):
        self.constrain_init = False
    if (self.deform_index == -1) or (self.deform_index == self.n_pts - 1):
        self.constrain_end = False
    constraints = coords
    constrain_indexes = [self.deform_index]
    if (self.constrain_init):
        constraints = np.vstack((constraints, self.org_traj[0]))
        constrain_indexes = np.vstack((constrain_indexes, [0]))
    if (self.constrain_end):
        constraints = np.vstack((constraints, self.org_traj[self.n_pts - 1]))
        constrain_indexes = np.vstack((constrain_indexes, [self.n_pts - 1]))
    
    if (DEBUG):
        print('Reproduction Constraints')
        print(constraints)
    
    #get reproduction
    reproduced_traj = self.algs[opt_alg_num](self.org_traj, constraints, constrain_indexes)
    if (DEBUG):
        print('Reproduction')
        print(reproduced_traj)
    
    #plot if requested
    if (plot):
        fig = plt.figure()
        if (self.n_dims == 1):
            plt.plot(reproduced_traj, COLORS[opt_alg_num])
            plt.plot(self.org_traj, 'k')
        elif (self.n_dims == 2):
            plt.plot(reproduced_traj[:, 0], reproduced_traj[:, 1], COLORS[opt_alg_num])
            plt.plot(self.org_traj[:, 0], self.org_traj[:, 1], 'k')
        elif (self.n_dims == 3):
            ax = fig.add_subplot(111, projection='3d')
            plt.plot(reproduced_traj[:, 0], reproduced_traj[:, 1], reproduced_traj[:, 2], COLORS[opt_alg_num])
            plt.plot(self.org_traj[:, 0], self.org_traj[:, 1], self.org_traj[:, 2], 'k')
        else:
            print('Unable to plot with current dimensions!')
        plt.show()
        plt.close('all')
    
    #return reproduced traj    
    return reproduced_traj
  
  #plots the similarity regions as taken from the classifier. Note: smaller subregions of similarity may be overwritten in the classifier by larger regions. To change this, you may want to change the type of classifier to a KNN classifier with a low K value. Note2: Does not work for >3D. Note3: This function is for general plotting. For plots that are more specific to the dimensionality of the demonstration, use plot_contour2D or plot_cube3D.
  #arguments
  #mode: a string, either 'save' or anything else. If 'save' the plot gets saved to the specified filepath. Otherwise it is shown to the user.
  #filepath: filepath to directory where plot is saved to
  #no returns
  def plot_classifier_results(self, mode='save', filepath=''):
    n_surf = self.grid_size * 3
    
    temp_deforms = []
    temp_index = ()
    
    for d in range(self.n_dims):
        temp_deforms.append(np.linspace(self.grid_vals[d][0], self.grid_vals[d][self.grid_size - 1], n_surf))
        temp_index = temp_index + (0,)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if (self.n_dims == 3):
        ax = fig.add_subplot(111, projection='3d')
    name = 'Classifier Results'    
    
    while (temp_index[0] < n_surf):
        plt_point = ()
        for d in range(self.n_dims):
            plt_point = plt_point + (temp_deforms[d][temp_index[d]],)
        if (DEBUG):
            print('Index')
            print(temp_index)
            print('Coordinates')
            print(plt_point)
        plt.plot(*zip(plt_point), COLORS[int(self.clf.predict(np.array([list(plt_point)])))] + '.')
        temp_index = list(temp_index)
        temp_index[self.n_dims - 1] = temp_index[self.n_dims - 1] + 1
        for d in reversed(range(self.n_dims)):
            if (temp_index[d] >= n_surf) and (d != 0):
                temp_index[d] = temp_index[d] - n_surf
                temp_index[d - 1] = temp_index[d - 1] + 1
        temp_index = tuple(temp_index)   
        
    if (mode == 'save'):
        plt.savefig(filepath + name + '.png')
    else:            
        plt.show() 
    plt.close('all')
  
  #plots the similarity regions as taken from the classifier using contours. Note: smaller subregions of similarity may be overwritten in the classifier by larger regions. To change this, you may want to change the type of classifier to a KNN classifier with a low K value. Note2: Only works for 2D. Note3: Colors may not be represented properly. See below.
  #arguments
  #mode: a string, either 'save' or anything else. If 'save' the plot gets saved to the specified filepath. Otherwise it is shown to the user.
  #filepath: filepath to directory where plot is saved to
  #no returns
  def plot_contour2D(self, mode='save', filepath=''):
    colors = ['r', 'g', 'b', 'c', 'm', 'y'] #contour doesn't care about the prediction value, so change these values to get the right color contours. Use plot_classifier_results to see the true colors.
    cm = LinearSegmentedColormap.from_list('my_colormap', COLORS[0:self.n_algs], N=self.n_algs)
    n_surf = self.grid_size * 5
    name = 'Similarity Contour'
    xnew = np.linspace(self.grid_vals[0][0], self.grid_vals[0][self.grid_size - 1], n_surf)
    ynew = np.linspace(self.grid_vals[1][0], self.grid_vals[1][self.grid_size - 1], n_surf)
    fig = plt.figure()
    xx, yy = np.meshgrid(xnew, ynew)
    Z = self.clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
    plt.plot(self.org_traj[0][0], self.org_traj[0][1], 'k*', markersize=30)
    if (mode == 'save'):
        plt.savefig(filepath + name + '.png')
    else:
        plt.show()
    plt.close('all')
  
  #plots the similarity regions as taken from the classifier in 3D. Note: smaller subregions of similarity may be overwritten in the classifier by larger regions. To change this, you may want to change the type of classifier to a KNN classifier with a low K value. Note2: Only works for 3D. Note3: Should be almost the same as the result from plot_classifier_results but with some changes to make the plot nicer.
  #arguments
  #mode: a string, either 'save' or anything else. If 'save' the plot gets saved to the specified filepath. Otherwise it is shown to the user.
  #filepath: filepath to directory where plot is saved to
  #no returns
  def plot_cube3D(self, mode='save', filepath=''):
    n_surf = self.grid_size * 5
    name = 'Similarity Region Cube'
    xnew = np.linspace(self.grid_vals[0][0], self.grid_vals[0][self.grid_size - 1], n_surf)
    ynew = np.linspace(self.grid_vals[1][0], self.grid_vals[1][self.grid_size - 1], n_surf)
    znew = np.linspace(self.grid_vals[2][0], self.grid_vals[2][self.grid_size - 1], n_surf)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for t in range (len(xnew)):
        for u in range (len(ynew)):
            for v in range (len(znew)):
                plt.plot(xnew[t], ynew[u], znew[v], COLORS[int(self.clf.predict(np.array([[xnew[t], ynew[u], znew[v]]])))] + '.', alpha=0.2)
    self.get_cube_outline(ax)
    if (mode == 'save'):
        plt.savefig(filepath + name + '.png')
    else:
        plt.show()
    plt.close('all')
     
  #shows the user the 2D similarity region and allows them to select a specific point from that region to be reproduced
  #arguments
  #plot (optional): if True, a plot of the reproduction will be shown to the user. If false, nothing happens.
  #returns the reproduced trajectory with greatest similarity.
  def reproduction_point_selection2D(self, plot=True):
    n_surf = self.grid_size * 5
    xnew = np.linspace(self.grid_vals[0][0], self.grid_vals[0][self.grid_size - 1], n_surf)
    ynew = np.linspace(self.grid_vals[1][0], self.grid_vals[1][self.grid_size - 1], n_surf)
    fig = plt.figure()
    for t in range(n_surf):
        for u in range(n_surf):
            plt.plot(xnew[t], ynew[u], COLORS[int(self.clf.predict(np.array([[xnew[t], ynew[u]]])))] + '.')
    plt.plot(self.org_traj[0][0], self.org_traj[0][1], 'k*', markersize=30)
    
    self.ix = 0.
    self.iy = 0.
    
    def onclick(event):
        self.ix, self.iy = event.xdata, event.ydata
    
        if self.ix < self.grid_vals[0][0] or self.ix > self.grid_vals[0][self.grid_size - 1] or self.iy < self.grid_vals[1][0] or self.iy > self.grid_vals[1][self.grid_size - 1]:
            print('Coordinate chosen is out of bounds, try again!')
        else:
            if (DEBUG):
                print('x = %f, y = %f'%(self.ix, self.iy))
                print('Coordinate chosen, no longer recording plot inputs')
            plt.plot(self.ix, self.iy, 'k+', markersize=20, mew=5.)
            fig.canvas.draw()
            time.sleep(2.)
            fig.canvas.mpl_disconnect(cid)
            plt.close('all')
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.show()
    
    if (DEBUG):
        print('x = %f, y = %f'%(self.ix, self.iy))
                
    return self.reproduce_at_point(coords=np.array([[self.ix, self.iy]]), plot=plot)
  
  #A previous version of reproduction_point_selection3D. For more user accessibility, please use that function. This function was kept as it is slightly faster and less buggy than reproduction_point_selection3D.
  def show_3d_in_2d_with_slider(self):
    self.gz_val = self.grid_vals[2][0]
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    n_surf = 40
    xnew = np.linspace(self.grid_vals[0][0], self.grid_vals[0][self.grid_size - 1], n_surf)
    ynew = np.linspace(self.grid_vals[1][0], self.grid_vals[1][self.grid_size - 1], n_surf)
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Select a new generalization point')
    
    l = [[plt.plot(xnew[i], ynew[j], COLORS[int(self.clf.predict(np.array([[xnew[i], ynew[j], self.gz_val]])))] + '.') for j in range(len(xnew))] for i in range(len(ynew))]
    
    axcolor = 'lightgoldenrodyellow'
    ax_z = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    
    s_z = Slider(ax_z, 'Z', self.grid_vals[2][0], self.grid_vals[2][self.grid_size - 1], self.gz_val)
    
    #plot cube seperately
    #colors = ['r', 'g', 'b', 'c', 'm', 'y']
    n_surf2 = self.grid_size
    #name = 'SVM Similarity Region_cube_tp'
    xnew2 = np.linspace(self.grid_vals[0][0], self.grid_vals[0][self.grid_size - 1], n_surf2)
    ynew2 = np.linspace(self.grid_vals[1][0], self.grid_vals[1][self.grid_size - 1], n_surf2)
    znew2 = np.linspace(self.grid_vals[2][0], self.grid_vals[2][self.grid_size - 1], n_surf2)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    #plt.tight_layout()
    for t in range (len(xnew2)):
        for u in range (len(ynew2)):
            for v in range (len(znew2)):
                ax2.scatter(xnew2[t], ynew2[u], znew2[v], c=COLORS[int(self.clf.predict(np.array([[xnew2[t], ynew2[u], znew2[v]]])))], alpha=0.5)
    self.get_cube_outline(ax2)
    xv = [self.grid_vals[0][0],self.grid_vals[0][0],self.grid_vals[0][self.grid_size - 1],self.grid_vals[0][self.grid_size - 1]]
    yv = [self.grid_vals[1][0],self.grid_vals[1][self.grid_size - 1],self.grid_vals[1][self.grid_size - 1],self.grid_vals[1][0]]
    zv = [self.gz_val,self.gz_val,self.gz_val,self.gz_val]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_alpha(0.1)
    poly.set_facecolor('w')
    ax2.add_collection3d(poly)    
    
    ax2.set_title('3D Similarity Region')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    self.slider_clicked = 0
    
    def update(val):
        self.gz_val = s_z.val
        print(self.gz_val)
        for i in range(len(xnew)):
            for j in range(len(ynew)):
                l[i][j][0].set_color(COLORS[int(self.clf.predict(np.array([[xnew[i], ynew[j], self.gz_val]])))])
        fig.canvas.draw_idle()
        ax2.collections.pop()
        zv = [self.gz_val,self.gz_val,self.gz_val,self.gz_val]
        verts = [list(zip(xv,yv,zv))]
        poly = Poly3DCollection(verts, linewidth=1)
        poly.set_edgecolor('k')
        poly.set_alpha(0.1)
        poly.set_facecolor('w')
        ax2.add_collection3d(poly)
        fig2.canvas.draw_idle()
        print('updated')
        self.slider_clicked = 1
        
    s_z.on_changed(update)
    
    self.coords = []

    def onclick(event):
        if (self.slider_clicked == 0):
            #global ix, iy
            ix, iy = event.xdata, event.ydata
        
            if ix < self.grid_vals[0][0] or ix > self.grid_vals[0][self.grid_size - 1] or iy < self.grid_vals[1][0] or iy > self.grid_vals[1][self.grid_size - 1]:
                print('Coordinate chosen is out of bounds, try again!')
                coords[:] = []
            else:
                #global coords
                self.coords.append(ix)
                self.coords.append(iy)
                print(self.coords)
                print('x = %f, y = %f'%(ix, iy))
                print('Coordinate chosen, no longer recording plot inputs')
                #plt.figure(fig.number)
                ax.plot(ix, iy, 'k+', markersize=20, mew=5.)
                fig.canvas.draw()
                time.sleep(2.)
                fig.canvas.mpl_disconnect(cid)
                plt.close('all')
        self.slider_clicked = 0
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.show()
    
    print(self.gz_val)
    self.coords.append(self.gz_val)
    print(self.coords)
    print(self.clf.predict(np.array([self.coords])))
    return self.reproduce_at_point([self.coords], plot=True)
    

  #shows the user the 3D similarity region and a 2D cross section and allows them to select a specific point from that cross section to be reproduced
  #arguments
  #plot (optional): if True, a plot of the reproduction will be shown to the user. If false, nothing happens.
  #returns the reproduced trajectory with greatest similarity.
  def reproduction_point_selection3D(self, plot=True):
  
    #set up default values (these values will be changed with slider/radio button updates
    self.gx_val = self.grid_vals[0][0]
    self.gy_val = self.grid_vals[1][0]
    self.gz_val = self.grid_vals[2][0]
    self.view = 'X'
    
    #plot cube seperately
    #colors = ['r', 'g', 'b', 'c', 'm', 'y']
    n_surf2 = self.grid_size
    #name = 'SVM Similarity Region_cube_tp'
    xnew2 = np.linspace(self.grid_vals[0][0], self.grid_vals[0][self.grid_size - 1], n_surf2)
    ynew2 = np.linspace(self.grid_vals[1][0], self.grid_vals[1][self.grid_size - 1], n_surf2)
    znew2 = np.linspace(self.grid_vals[2][0], self.grid_vals[2][self.grid_size - 1], n_surf2)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    #plt.tight_layout()
    for t in range (len(xnew2)):
        for u in range (len(ynew2)):
            for v in range (len(znew2)):
                ax2.scatter(xnew2[t], ynew2[u], znew2[v], c=COLORS[int(self.clf.predict(np.array([[xnew2[t], ynew2[u], znew2[v]]])))], alpha=0.5)
    self.get_cube_outline(ax2)
    xv = [self.grid_vals[0][0],self.grid_vals[0][0],self.grid_vals[0][self.grid_size - 1],self.grid_vals[0][self.grid_size - 1]]
    yv = [self.grid_vals[1][0],self.grid_vals[1][self.grid_size - 1],self.grid_vals[1][self.grid_size - 1],self.grid_vals[1][0]]
    zv = [self.gz_val,self.gz_val,self.gz_val,self.gz_val]
    verts = [list(zip(xv,yv,zv))]
    poly = Poly3DCollection(verts, linewidth=1)
    poly.set_edgecolor('k')
    poly.set_alpha(0.1)
    poly.set_facecolor('w')
    ax2.add_collection3d(poly)    
    
    ax2.set_title('3D Similarity Region')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    #set up point selection plot
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    n_surf = 40
    xnew = np.linspace(self.grid_vals[0][0], self.grid_vals[0][self.grid_size - 1], n_surf)
    ynew = np.linspace(self.grid_vals[1][0], self.grid_vals[1][self.grid_size - 1], n_surf)
    znew = np.linspace(self.grid_vals[2][0], self.grid_vals[2][self.grid_size - 1], n_surf)
    
    fig, ax = plt.subplots()
    
    plt.subplots_adjust(left=0.25, bottom=0.3)
    
    #set up slider positions and values
    axcolor = 'lightgoldenrodyellow'
    ax_x = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_y = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    ax_z = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    
    s_x = Slider(ax_x, 'X', self.grid_vals[0][0], self.grid_vals[0][self.grid_size - 1], self.gx_val)
    s_y = Slider(ax_y, 'Y', self.grid_vals[1][0], self.grid_vals[1][self.grid_size - 1], self.gy_val)
    s_z = Slider(ax_z, 'Z', self.grid_vals[2][0], self.grid_vals[2][self.grid_size - 1], self.gz_val)
    
    #set up radio buttons to switch views
    rax = plt.axes([0.05, 0.7, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, ('X', 'Y', 'Z'))
    
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_title('Select a new generalization point')
    
    #store each plotted point so they can be updated later
    self.l = [[ax.plot(ynew[i], znew[j], COLORS[int(self.clf.predict(np.array([[self.gx_val, ynew[i], znew[j]]])))] + '.') for j in range(len(znew))] for i in range(len(ynew))]
    
    
    
    #so that clicking the slider doesn't register as clicking on the grid
    self.slider_clicked = 0
    
    #set up slider update functions
    def update_x(val):
        self.gx_val = s_x.val
        print(self.gx_val)
        for i in range(len(ynew)):
            for j in range(len(znew)):
                self.l[i][j][0].set_color(COLORS[int(self.clf.predict(np.array([[self.gx_val, ynew[i], znew[j]]])))])
        fig.canvas.draw_idle()
        ax2.collections.pop()
        xv = [self.gx_val,self.gx_val,self.gx_val,self.gx_val]
        yv = [self.grid_vals[1][0],self.grid_vals[1][self.grid_size - 1],self.grid_vals[1][self.grid_size - 1],self.grid_vals[1][0]]
        zv = [self.grid_vals[2][0],self.grid_vals[2][0],self.grid_vals[2][self.grid_size - 1],self.grid_vals[2][self.grid_size - 1]]
        verts = [list(zip(xv,yv,zv))]
        poly = Poly3DCollection(verts, linewidth=1)
        poly.set_edgecolor('k')
        poly.set_alpha(0.1)
        poly.set_facecolor('w')
        ax2.add_collection3d(poly)
        fig2.canvas.draw_idle()
        print('updated')
        self.slider_clicked = 1
        
    s_x.on_changed(update_x)
    
    def update_y(val):
        self.gy_val = s_y.val
        print(self.gy_val)
        for i in range(len(xnew)):
            for j in range(len(znew)):
                self.l[i][j][0].set_color(COLORS[int(self.clf.predict(np.array([[xnew[i], self.gy_val, znew[j]]])))])
        fig.canvas.draw_idle()
        ax2.collections.pop()
        xv = [self.grid_vals[0][0],self.grid_vals[0][self.grid_size - 1],self.grid_vals[0][self.grid_size - 1],self.grid_vals[0][0]]
        yv = [self.gy_val,self.gy_val,self.gy_val,self.gy_val]
        zv = [self.grid_vals[2][0],self.grid_vals[2][0],self.grid_vals[2][self.grid_size - 1],self.grid_vals[2][self.grid_size - 1]]
        verts = [list(zip(xv,yv,zv))]
        poly = Poly3DCollection(verts, linewidth=1)
        poly.set_edgecolor('k')
        poly.set_alpha(0.1)
        poly.set_facecolor('w')
        ax2.add_collection3d(poly)
        fig2.canvas.draw_idle()
        print('updated')
        self.slider_clicked = 1
        
    s_y.on_changed(update_y)
    
    def update_z(val):
        self.gz_val = s_z.val
        print(self.gz_val)
        for i in range(len(xnew)):
            for j in range(len(ynew)):
                self.l[i][j][0].set_color(COLORS[int(self.clf.predict(np.array([[xnew[i], ynew[j], self.gz_val]])))])
        fig.canvas.draw_idle()
        ax2.collections.pop()
        xv = [self.grid_vals[0][0],self.grid_vals[0][0],self.grid_vals[0][self.grid_size - 1],self.grid_vals[0][self.grid_size - 1]]
        yv = [self.grid_vals[1][0],self.grid_vals[1][self.grid_size - 1],self.grid_vals[1][self.grid_size - 1],self.grid_vals[1][0]]
        zv = [self.gz_val,self.gz_val,self.gz_val,self.gz_val]
        verts = [list(zip(xv,yv,zv))]
        poly = Poly3DCollection(verts, linewidth=1)
        poly.set_edgecolor('k')
        poly.set_alpha(0.1)
        poly.set_facecolor('w')
        ax2.add_collection3d(poly)
        fig2.canvas.draw_idle()
        print('updated')
        self.slider_clicked = 1
        
    s_z.on_changed(update_z)
    
    
    def switch_view(label):
        self.view = label
        self.slider_clicked = 1
        if (self.view == 'X'):
            for i in range(len(ynew)):
                for j in range(len(znew)):
                    self.l[i][j][0].remove()
            self.l = [[ax.plot(ynew[i], znew[j], COLORS[int(self.clf.predict(np.array([[self.gx_val, ynew[i], znew[j]]])))] + '.') for j in range(len(znew))] for i in range(len(ynew))]
            update_x(self.gx_val)
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            ax.set_xlim([self.grid_vals[1][0], self.grid_vals[1][self.grid_size - 1]])
            ax.set_ylim([self.grid_vals[2][0], self.grid_vals[2][self.grid_size - 1]])
        if (self.view == 'Y'):
            for i in range(len(xnew)):
                for j in range(len(znew)):
                    self.l[i][j][0].remove()
            self.l = [[ax.plot(xnew[i], znew[j], COLORS[int(self.clf.predict(np.array([[xnew[i], self.gy_val, znew[j]]])))] + '.') for j in range(len(znew))] for i in range(len(xnew))]
            update_y(self.gy_val)
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_xlim([self.grid_vals[0][0], self.grid_vals[0][self.grid_size - 1]])
            ax.set_ylim([self.grid_vals[2][0], self.grid_vals[2][self.grid_size - 1]])
        if (self.view == 'Z'):
            for i in range(len(xnew)):
                for j in range(len(ynew)):
                    self.l[i][j][0].remove()
            self.l = [[ax.plot(xnew[i], ynew[j], COLORS[int(self.clf.predict(np.array([[xnew[i], ynew[j], self.gz_val]])))] + '.') for j in range(len(ynew))] for i in range(len(xnew))]
            update_y(self.gy_val)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim([self.grid_vals[1][0], self.grid_vals[0][self.grid_size - 1]])
            ax.set_ylim([self.grid_vals[0][0], self.grid_vals[1][self.grid_size - 1]])
        plt.draw()
        if (DEBUG):
            print(self.view)
            
    radio.on_clicked(switch_view)
    
    #what to do when point is selected
    self.coords = []

    def onclick(event):
        if (self.slider_clicked == 0):
            if (DEBUG):
                print(self.view)
                print(self.grid_vals)
                
            ix, iy = event.xdata, event.ydata
            print('ix = %f, iy = %f'%(ix, iy))
        
            if (self.view == 'X') and (ix < self.grid_vals[1][0] or ix > self.grid_vals[1][self.grid_size - 1] or iy < self.grid_vals[2][0] or iy > self.grid_vals[2][self.grid_size - 1]):
                print('Coordinate chosen is out of bounds, try again!')
                self.coords[:] = []
            elif (self.view == 'Y') and (ix < self.grid_vals[0][0] or ix > self.grid_vals[0][self.grid_size - 1] or iy < self.grid_vals[2][0] or iy > self.grid_vals[2][self.grid_size - 1]):
                print('Coordinate chosen is out of bounds, try again!')
                self.coords[:] = []
            elif (self.view == 'Z') and (ix < self.grid_vals[0][0] or ix > self.grid_vals[0][self.grid_size - 1] or iy < self.grid_vals[1][0] or iy > self.grid_vals[1][self.grid_size - 1]):
                print('Coordinate chosen is out of bounds, try again!')
                self.coords[:] = []
            else:
                #global coords
                if (self.view == 'X'):
                    self.coords.append(self.gx_val)
                self.coords.append(ix)
                if (self.view == 'Y'):
                    self.coords.append(self.gy_val)
                self.coords.append(iy)
                if (self.view == 'Z'):
                    self.coords.append(self.gz_val)
                print(self.coords)
                print('Coordinate chosen, no longer recording plot inputs')
                #plt.figure(fig.number)
                ax.plot(ix, iy, 'k+', markersize=20, mew=5.)
                fig.canvas.draw()
                fig.canvas.mpl_disconnect(cid)
                time.sleep(2.)
                plt.close('all')
        self.slider_clicked = 0
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    #show the plot to allow for point selection
    plt.show()
    
    if (DEBUG):
        print(self.coords)
        print(self.clf.predict(np.array([self.coords])))
    
    #get reproduction at selected point
    return self.reproduce_at_point([self.coords], plot=True)
        
        
def main3D():
    #simple example of 3D use
    x = np.linspace(1, 10)
    y = np.linspace(1, 10)**2
    z = np.linspace(1, 10)**3
    
    my_mlfd = metalfd()
    my_mlfd.add_traj(np.transpose(np.vstack((x, y, z))))
    import ja
    my_mlfd.add_representation(ja.perform_ja_general, 'JA')
    import lte
    my_mlfd.add_representation(lte.LTE_ND_any_constraints, 'LTE')
    import sys
    # insert at 1, 0 is the script path (or '' in REPL)
    sys.path.insert(1, './dmp_pastor_2009/')
    import perform_dmp as dmp
    my_mlfd.add_representation(dmp.perform_dmp_general, 'DMP')
    import similaritymeasures
    my_mlfd.add_metric(similaritymeasures.frechet_dist, is_dissim=True)
    
    my_mlfd.create_grid(plot=False)
    
    my_mlfd.deform_traj()
    
    my_mlfd.calc_similarities()
    my_mlfd.plot_heatmap(mode='show')
    
    my_mlfd.save_to_h5(filename='../data/test3d.h5')
    #my_mlfd.load_from_h5(filename='../data/test3d.h5')
    
    my_mlfd.get_strongest_sims(0.2)
    my_mlfd.plot_strongest_sims(mode='show')
    
    my_mlfd.set_up_classifier()
    
    my_mlfd.plot_classifier_results(mode='show')
    my_mlfd.plot_cube3D(mode='show')
    my_mlfd.reproduction_point_selection3D()
        
def main2D():
    #simple example of 2D use
    x = np.linspace(1, 10, 1000)
    y = np.linspace(1, 10, 1000)**2
    
    my_mlfd = metalfd()
    my_mlfd.add_traj(np.transpose(np.vstack((x, y))))
    import ja
    my_mlfd.add_representation(ja.perform_ja_general, 'JA')
    import lte
    my_mlfd.add_representation(lte.LTE_ND_any_constraints, 'LTE')
    import sys
    # insert at 1, 0 is the script path (or '' in REPL)
    sys.path.insert(1, './dmp_pastor_2009/')
    import perform_dmp as dmp
    my_mlfd.add_representation(dmp.perform_dmp_general, 'DMP')
    import similaritymeasures
    my_mlfd.add_metric(similaritymeasures.frechet_dist, is_dissim=True)
    
    my_mlfd.create_grid(plot=False)
    
    my_mlfd.deform_traj(plot=True)
    
    my_mlfd.calc_similarities()
    my_mlfd.plot_heatmap(mode='show')
    
    my_mlfd.save_to_h5(filename='../data/test.h5')
    #my_mlfd.load_from_h5(filename='../data/test.h5')
    
    my_mlfd.get_strongest_sims(0.2)
    my_mlfd.plot_strongest_sims(mode='show')
    
    my_mlfd.set_up_classifier()
    
    my_mlfd.plot_classifier_results(mode='show')
    my_mlfd.plot_contour2D(mode='show')
    my_mlfd.reproduction_point_selection2D()
        
if __name__ == '__main__':
  main2D()    
        