import mlfd
import numpy as np
import h5py
import preprocess
import ja
import lte
import similaritymeasures
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import demo_to_ur_ws
import random

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/BH/Documents/GitHub/Pearl-ur5e_matlab-implementations/python deformations/dmp_for_comparison/')

import perform_new_dmp as pnd

fnames = ['36_56']

fname_start = '../h5 files/mlfd_demo/bad preprocessed recorded_demo Fri Jul  3 14_'
fname_end = ' 2020.h5'

def get_xyz_demo(filename):
    #open the file
    hf = h5py.File(filename, 'r')
    #navigate to necessary data and store in numpy arrays
    demo = hf.get('demo1')
    tf_info = demo.get('tf_info')
    pos_rot_data = tf_info.get('pos_rot_data')
    pos_rot_data = np.array(pos_rot_data)
    #close out file
    hf.close()
    x = pos_rot_data[0]
    y = pos_rot_data[1]
    z = pos_rot_data[2]
    return [x, y, z]

def main():
    global fnames
    global fname_start
    global fname_end
    for i in range(len(fnames)):
        full_name = fname_start + fnames[i] + fname_end
        [x, y, z] = get_xyz_demo(full_name)
    
        plt_fpath = '3d_reproduction_testing_from_demo/' + fnames[i] + '/'
        try:
            os.makedirs(plt_fpath)
        except OSError:
            print ("Creation of the directory %s failed" % plt_fpath)
        else:
            print ("Successfully created the directory %s" % plt_fpath)
            
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.plot(x, y, z, 'k', linewidth=10)
        #ax.plot([x[0]], [y[0]], [z[0]],  'k*', markersize=40)
        #ax.plot([x[-1]], [y[-1]], [z[-1]],  'k.', markersize=40)
        #ax.set_facecolor('white')
        ##plt.axis('off')
        ##ax.set_title('Demonstration')
        ##ax.set_xlabel('X')
        ##ax.set_ylabel('Y')
        ##ax.set_zlabel('Z')
        ##ax.view_init(elev=1., azim=-89.)
        #plt.savefig(plt_fpath + 'demonstration_alone_axis.png')
        #plt.close('all')
        #input()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, 'k', linewidth=5)
        ax.plot([x[0]], [y[0]], [z[0]],  'k*', markersize=20)
        ax.plot([x[-1]], [y[-1]], [z[-1]],  'k.', markersize=20)
        #ax.set_facecolor('white')
        #plt.axis('off')
        ax.set_title('Demonstration')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=1., azim=-89.)
        plt.savefig(plt_fpath + 'demonstration_axis.png')
        plt.close('all')
        
        ### set up mlfd ##
        #my_mlfd = mlfd.mlfd()
        #my_mlfd.add_traj_dimension(x, 'x')
        #my_mlfd.add_traj_dimension(y, 'y')
        #my_mlfd.add_traj_dimension(z, 'z')
        #my_mlfd.add_deform_alg(ja.perform_ja_improved, 'JA')
        #my_mlfd.add_deform_alg(lte.perform_lte_improved, 'LTE')
        #my_mlfd.add_deform_alg(pnd.perform_new_dmp_adapted, 'DMP')
        #my_mlfd.add_metric(similaritymeasures.frechet_dist, name='Frechet', is_dissim=True)
        #
        ### first run params ##
        #my_mlfd.create_grid()
        #my_mlfd.deform_traj()
        #plt.close('all')
        #my_mlfd.calc_metrics()
        #
        ### save/load from file ##
        #my_mlfd.save_results(plt_fpath + fnames[i] + '_mlfd.h5')
        ##my_mlfd.read_from_h5(plt_fpath + fnames[i] + '_mlfd.h5')
        #
        ### interpret results ##
        #my_mlfd.set_up_classifier()
        #my_mlfd.generate_svm_region(mode='save', filepath=plt_fpath)
        ##my_mlfd.get_image_slices(mode='save', filepath=plt_fpath)
        #my_mlfd.show_3d_similarity_optimal(mode='save', filepath=plt_fpath)
        ##my_mlfd.generate_separated_svm_region(mode='save', filepath=plt_fpath)
        #my_mlfd.generate_svm_region_cube(mode='save', filepath=plt_fpath)
        ##my_mlfd.reproduce_optimal_at_point([[0.257, -0.46949, 0.57821]], plot=True, mode='save', filepath=plt_fpath)
        ##my_mlfd.reproduce_optimal_at_point([[0.19683, -0.52011, 0.62280]], plot=True, mode='save', filepath=plt_fpath)
        ##my_mlfd.reproduce_optimal_at_point([[0.24651, -0.45814, 0.61672]], plot=True, mode='save', filepath=plt_fpath)
        ##my_mlfd.show_3d_in_2d_with_slider()
        ##my_mlfd.show_3d_in_2d_with_slider()
        #
        ##dist = my_mlfd.get_demo_dist()
        ##alg_repros = np.zeros((3))
        ##random.seed()
        ##fp = h5py.File(plt_fpath + '/' + fnames[i] + '_' + '_3D_reproduction.h5', 'w')
        ##fp.create_dataset('org/x', data=x)
        ##fp.create_dataset('org/y', data=y)
        ##fp.create_dataset('org/z', data=z)
        ##i = 0
        ##colors = ['r', 'g', 'b']
        ##while not (alg_repros.all() == 1):
        ##    x_offset = random.uniform(-dist / 8., dist / 8.)
        ##    y_offset = random.uniform(-dist / 8., dist / 8.)
        ##    z_offset = random.uniform(-dist / 8., dist / 8.) 
        ##    [opt_x, opt_y, opt_z, alg_name] = my_mlfd.reproduce_optimal_at_point([[x[0] + x_offset, y[0] + y_offset, z[0] + z_offset]], plot=False, mode='save', filepath=plt_fpath) 
        ##    if (alg_name == 'JA'):
        ##        cur_alg = 0
        ##    if (alg_name == 'LTE'):
        ##        cur_alg = 1
        ##    if (alg_name == 'DMP'):
        ##        cur_alg = 2
        ##    if not (alg_repros[cur_alg] == 1):
        ##        fig = plt.figure()
        ##        ax = fig.add_subplot(111, projection='3d')
        ##        ax.plot(x, y, z, 'k', linewidth=10)
        ##        ax.plot([x[0]], [y[0]], [z[0]],  'k*', markersize=40)
        ##        ax.plot([x[-1]], [y[-1]], [z[-1]],  'k.', markersize=40)
        ##        ax.plot(opt_x, opt_y, opt_z, colors[cur_alg], linewidth=10)
        ##        ax.plot([opt_x[0]], [opt_y[0]], [opt_z[0]],  'k+', markersize=30, mew=7)
        ##        ax.set_facecolor('white')
        ##        plt.axis('off')
        ##        plt.savefig(plt_fpath + 'demonstration_' + alg_name + '_reproduction.png')
        ##        plt.close('all')
        ##        fp.create_dataset(alg_name + '/x', data=opt_x)
        ##        fp.create_dataset(alg_name + '/y', data=opt_y)
        ##        fp.create_dataset(alg_name + '/z', data=opt_z)
        ##        alg_repros[cur_alg] = 1.
        ##    i = i + 1
        ##    print(i)
        ##fp.close()
     
if __name__ == '__main__':
  main()