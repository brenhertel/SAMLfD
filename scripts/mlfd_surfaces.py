# surface plotting of similarity

import mlfd
import numpy as np
import h5py
import ja
import lte
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './dmp_pastor_2009/')
import perform_dmp as dmp
import similaritymeasures
import os
import douglas_peucker as dp
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
mpl.rcParams.update({'font.size': 48})

#lasa_names = ['Angle','BendedLine','CShape','DoubleBendedLine','GShape', \
#              'heee','JShape','JShape_2','Khamesh','Leaf_1', \
#              'Leaf_2','Line','LShape','NShape','PShape', \
#              'RShape','Saeghe','Sharpc','Sine','Snake', \
#              'Spoon','Sshape','Trapezoid','Worm','WShape', \
#              'Zshape','Multi_Models_1','Multi_Models_2','Multi_Models_3','Multi_Models_4']
 
lasa_names = ['Saeghe']
 
def get_lasa_traj1(shape_name):
    #ask user for the file which the playback is for
    #filename = raw_input('Enter the filename of the .h5 demo: ')
    #open the file
    filename = '../data/lasa_dataset.h5'
    hf = h5py.File(filename, 'r')
    #navigate to necessary data and store in numpy arrays
    shape = hf.get(shape_name)
    demo = shape.get('demo1')
    pos_info = demo.get('pos')
    pos_data = np.array(pos_info)
    y_data = np.delete(pos_data, 0, 1)
    x_data = np.delete(pos_data, 1, 1)
    #close out file
    hf.close()
    return [x_data, y_data]

def my_hd(u, v):
    return directed_hausdorff(u, v)[0]

def main():
    ## set up demonstration
    global lasa_names
    for lname in range (len(lasa_names)):
        lasa_name = lasa_names[lname]
        print(lasa_name)
        plt_fpath = '../example_outputs/surface_ex_hd/' + lasa_name + '/'
        try:
            os.makedirs(plt_fpath)
        except OSError:
            print ("Creation of the directory %s failed" % plt_fpath)
        else:
            print ("Successfully created the directory %s" % plt_fpath)
        [x, y] = get_lasa_traj1(lasa_name)
        traj = np.hstack((x, y))
        
        plt.figure()
        plt.axis('off')
        
        plt.subplot(111)
        plt.xticks([])
        plt.yticks([])
        plt.plot(x, y, 'k', lw=7)
        plt.plot(x[0], y[0], 'k*', lw=7, ms=25)
        plt.plot(x[-1], y[-1], 'k.', lw=7, ms=25)
        #plt.axis('off')
        plt.savefig(plt_fpath + 'demo.png', dpi=300)
        plt.close('all')
        
        #traj = dp.DouglasPeuckerPoints(traj, 100)
        #
        ### set up SAMLfD Object
        #my_mlfd = mlfd.SAMLfD()
        #my_mlfd.add_traj(traj)
        #my_mlfd.add_representation(ja.perform_ja_general, 'JA')
        #my_mlfd.add_representation(lte.LTE_ND_any_constraints, 'LTE')
        #my_mlfd.add_representation(dmp.perform_dmp_general, 'DMP')
        #my_mlfd.add_metric(my_hd, is_dissim=True)
        ##my_mlfd.plot_org_traj(mode='save', filepath=plt_fpath)
        ##
        #### create meshgrid
        ##my_mlfd.create_grid(plot=False) #use default values for grid creation (defaults include initial point deformation)
        ##
        #### deform at each point on grid
        ##my_mlfd.deform_traj(plot=False)
        ##
        #### calculate similarities of deformations
        ##my_mlfd.calc_similarities()
        ##my_mlfd.plot_heatmap(mode='save', filepath=plt_fpath)
        ##
        #### save/load results
        ##my_mlfd.save_to_h5(plt_fpath + 'mlfd_' + lasa_name + '.h5')
        #my_mlfd.load_from_h5(plt_fpath + 'mlfd_' + lasa_name + '.h5')
        #
        #
        ### set up classifier
        #my_mlfd.get_strongest_sims(0.1)
        #my_mlfd.set_up_classifier()
        #
        ### get similarity region & reproductions
        ##my_mlfd.plot_classifier_results(mode='save', filepath=plt_fpath)
        ##my_mlfd.plot_contour2D(mode='save', filepath=plt_fpath)
        #
        #sim_vals = my_mlfd.sim_vals
        #rep_sim_vals = []
        #
        #
        #cmaps = [cm.Reds, cm.Greens, cm.Blues]
        #
        #for r in range(my_mlfd.n_algs):
        #    #representation_sims = np.zeros((my_mlfd.grid_size, my_mlfd.grid_size))
        #    #for gd_pt in range(my_mlfd.grid_size**my_mlfd.n_dims):
        #    #    ind = my_mlfd.convert_gd_pt_to_index(gd_pt)
        #    #    representation_sims[ind] = sim_vals[gd_pt][r]
        #    #np.savetxt(plt_fpath + 'sim_arr' + str(r) + '.txt', representation_sims)
        #    representation_sims = np.loadtxt(plt_fpath + 'sim_arr' + str(r) + '.txt')
        #    rep_sim_vals.append(representation_sims)
        #    fig = plt.figure()
        #    fig.patch.set_facecolor('white')
        #    ax = plt.axes(projection='3d')
        #    ax.patch.set_facecolor('white')
        #    ax.xaxis._axinfo["grid"]['color'] = 'k'
        #    ax.yaxis._axinfo["grid"]['color'] = 'k'
        #    ax.zaxis._axinfo["grid"]['color'] = 'k'
        #    ax.xaxis.pane.fill = False
        #    ax.yaxis.pane.fill = False
        #    ax.zaxis.pane.fill = False
        #    ax.xaxis.pane.set_edgecolor('w')
        #    ax.yaxis.pane.set_edgecolor('w')
        #    ax.zaxis.pane.set_edgecolor('w')
        #    ax.set_xticklabels([])
        #    ax.set_yticklabels([])
        #    ax.set_zticklabels([])
        #    X = np.arange(0, my_mlfd.grid_size)
        #    Y = np.arange(0, my_mlfd.grid_size)
        #    X, Y = np.meshgrid(X, Y)
        #    surf = ax.plot_surface(X, Y, representation_sims, cmap=cmaps[r])
        #    plt.savefig(plt_fpath + 'surface' + str(r) + '_clean.png', dpi=300)
        #    #plt.show()
        #    plt.close('all')
        #    
        #best_sim_vals = np.zeros((my_mlfd.grid_size, my_mlfd.grid_size))
        #for i in range(my_mlfd.grid_size):
        #    for j in range(my_mlfd.grid_size):
        #        best_sim_vals[i, j] = max([rep_sim_vals[r][i, j] for r in range(my_mlfd.n_algs)])
        #        
        #for r in range(my_mlfd.n_algs):
        #    sum = 0.
        #    for i in range(my_mlfd.grid_size):
        #        for j in range(my_mlfd.grid_size):
        #            sum += best_sim_vals[i, j] - rep_sim_vals[r][i, j]
        #    print('total difference between representation %d and mlfd surface: %f' % (r, sum))
        #    
        #fig = plt.figure()
        #fig.patch.set_facecolor('white')
        #ax = plt.axes(projection='3d')
        #ax.patch.set_facecolor('white')
        #ax.xaxis._axinfo["grid"]['color'] = 'k'
        #ax.yaxis._axinfo["grid"]['color'] = 'k'
        #ax.zaxis._axinfo["grid"]['color'] = 'k'
        ##ax.set_xlabel('x')
        ##ax.set_ylabel('y')
        ##ax.set_zlabel('similarity')
        #ax.xaxis.pane.fill = False
        #ax.yaxis.pane.fill = False
        #ax.zaxis.pane.fill = False
        #ax.xaxis.pane.set_edgecolor('w')
        #ax.yaxis.pane.set_edgecolor('w')
        #ax.zaxis.pane.set_edgecolor('w')
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        #ax.set_zticklabels([])
        #
        #X = np.arange(0, my_mlfd.grid_size)
        #Y = np.arange(0, my_mlfd.grid_size)
        #X, Y = np.meshgrid(X, Y)
        #surf = ax.plot_surface(X, Y, rep_sim_vals[0], cmap=cmaps[0])
        #surf = ax.plot_surface(X, Y, rep_sim_vals[1], cmap=cmaps[1])
        #surf = ax.plot_surface(X, Y, rep_sim_vals[2], cmap=cmaps[2])
        #surf = ax.plot_surface(X, Y, best_sim_vals, cmap=cm.Purples, alpha=1.0)
        #plt.savefig(plt_fpath + 'all_surface_clean.png', dpi=300)
        #plt.close('all')
        #print(lasa_name)
        #print('----------')
        
     
if __name__ == '__main__':
  main()