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


def main():
    ## set up demonstration
    
    skill = 'REACHING'
    filename = '../data/' + skill + '_dataset.h5'
    #open the file
    hf = h5py.File(filename, 'r')
    
    fnum = 1
    fname = 'file' + str(fnum)
    print(fname)
    f = hf.get(fname)
    
    dnum = 4
    dname = 'demo' + str(dnum) 
    print(dname)
    d = f.get(dname)
    pos = d.get('pos')
    
    pos_arr = np.array(pos)
    traj = np.transpose(pos_arr[0:3])
    
    ## set up save data filepath 
    plt_fpath = '../example_outputs/3d_initpt_example/'
    try:
        os.makedirs(plt_fpath)
    except OSError:
        print ("Creation of the directory %s failed" % plt_fpath)
    else:
        print ("Successfully created the directory %s" % plt_fpath)
      
    ## set up SAMLfD Object
    my_mlfd = mlfd.SAMLfD()
    my_mlfd.add_traj(traj)
    my_mlfd.add_representation(ja.perform_ja_general, 'JA')
    my_mlfd.add_representation(lte.LTE_ND_any_constraints, 'LTE')
    my_mlfd.add_representation(dmp.perform_dmp_general, 'DMP')
    my_mlfd.add_metric(similaritymeasures.frechet_dist, is_dissim=True)
    my_mlfd.plot_org_traj(mode='save', filepath=plt_fpath)
    
    ## create meshgrid
    my_mlfd.create_grid(plot=True) #index defaults to initial point
    
    ## deform at each point on grid
    my_mlfd.deform_traj()
    
    ## calculate similarities of deformations
    my_mlfd.calc_similarities()
    my_mlfd.plot_heatmap(mode='save', filepath=plt_fpath)
    
    ## save/load results
    my_mlfd.save_to_h5(plt_fpath + 'mlfd_' + skill + str(fnum) + str(dnum) + '.h5')
    #my_mlfd.load_from_h5(plt_fpath + 'mlfd_' + skill + str(fnum) + str(dnum) + '.h5')
    
    
    ## set up classifier
    my_mlfd.get_strongest_sims(0.1)
    my_mlfd.set_up_classifier()
    
    ## get similarity region & reproductions
    my_mlfd.plot_classifier_results(mode='save', filepath=plt_fpath)
    my_mlfd.plot_cube3D(mode='save', filepath=plt_fpath)
    my_mlfd.reproduction_point_selection3D()
     
     
if __name__ == '__main__':
  main()