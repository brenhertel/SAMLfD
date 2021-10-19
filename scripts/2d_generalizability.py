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

def main():
    ## set up demonstration
    lasa_name = 'Saeghe'
    plt_fpath1 = '../example_outputs/2d_generalize_example1/'
    plt_fpath2 = '../example_outputs/2d_generalize_example2/'
    try:
        os.makedirs(plt_fpath1)
    except OSError:
        print ("Creation of the directory %s failed" % plt_fpath1)
    else:
        print ("Successfully created the directory %s" % plt_fpath1)
    try:
        os.makedirs(plt_fpath2)
    except OSError:
        print ("Creation of the directory %s failed" % plt_fpath2)
    else:
        print ("Successfully created the directory %s" % plt_fpath2)
    [x, y] = get_lasa_traj1(lasa_name)
    traj = np.hstack((x, y))
    
    traj = dp.DouglasPeuckerPoints(traj, 100)
    
    ## set up SAMLfD Object
    my_mlfd1 = mlfd.SAMLfD()
    my_mlfd1.add_traj(traj)
    my_mlfd1.add_representation(ja.perform_ja_general, 'JA')
    my_mlfd1.add_representation(lte.LTE_ND_any_constraints, 'LTE')
    my_mlfd1.add_representation(dmp.perform_dmp_general, 'DMP')
    my_mlfd1.add_metric(similaritymeasures.frechet_dist, is_dissim=True)
    my_mlfd1.plot_org_traj(mode='save', filepath=plt_fpath1)
    
    my_mlfd2 = mlfd.SAMLfD()
    my_mlfd2.add_traj(traj)
    my_mlfd2.add_representation(dmp.perform_dmp_general, 'DMP')
    my_mlfd2.add_metric(similaritymeasures.frechet_dist, is_dissim=True)
    my_mlfd2.plot_org_traj(mode='save', filepath=plt_fpath2)
    
    ## create meshgrid
    my_mlfd1.create_grid(plot=False) #use default values for grid creation (defaults include initial point deformation)
    my_mlfd2.create_grid(plot=False) #use default values for grid creation (defaults include initial point deformation)
    
    ## deform at each point on grid
    my_mlfd1.deform_traj(plot=False)
    my_mlfd2.deform_traj(plot=False)
    
    ## calculate similarities of deformations
    my_mlfd1.calc_similarities()
    my_mlfd1.plot_heatmap(mode='save', filepath=plt_fpath1)
    my_mlfd2.calc_similarities()
    my_mlfd2.plot_heatmap(mode='save', filepath=plt_fpath2)
    
    ## save/load results
    my_mlfd1.save_to_h5(plt_fpath1 + 'mlfd_' + lasa_name + '.h5')
    #my_mlfd1.load_from_h5(plt_fpath1 + 'mlfd_' + lasa_name + '.h5')
    my_mlfd2.save_to_h5(plt_fpath2 + 'mlfd_' + lasa_name + '.h5')
    #my_mlfd2.load_from_h5(plt_fpath2 + 'mlfd_' + lasa_name + '.h5')
    
    
    ## set up classifier
    my_mlfd1.get_strongest_sims(0.5)
    my_mlfd1.set_up_classifier()
    my_mlfd2.get_strongest_sims(0.5)
    my_mlfd2.set_up_classifier()
    
    ## get similarity region & reproductions
    my_mlfd1.plot_classifier_results(mode='save', filepath=plt_fpath1)
    my_mlfd1.plot_contour2D(mode='save', filepath=plt_fpath1)
    #my_mlfd1.reproduction_point_selection2D()
    
    my_mlfd2.plot_classifier_results(mode='save', filepath=plt_fpath2)
    my_mlfd2.plot_contour2D(mode='save', filepath=plt_fpath2)
    #my_mlfd.reproduction_point_selection2D()
     
if __name__ == '__main__':
  main()