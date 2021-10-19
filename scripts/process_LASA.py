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

lasa_names = ['Angle','BendedLine','CShape','DoubleBendedLine','GShape', \
              'heee','JShape','JShape_2','Khamesh','Leaf_1', \
              'Leaf_2','Line','LShape','NShape','PShape', \
              'RShape','Saeghe','Sharpc','Sine','Snake', \
              'Spoon','Sshape','Trapezoid','Worm','WShape', \
              'Zshape','Multi_Models_1','Multi_Models_2','Multi_Models_3','Multi_Models_4']

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
    global lasa_names
    for i in range (len(lasa_names)):
        plt_fpath = 'lasa_mlfd/' + lasa_names[i] + '/'
        try:
            os.makedirs(plt_fpath)
        except OSError:
            print ("Creation of the directory %s failed" % plt_fpath)
        else:
            print ("Successfully created the directory %s" % plt_fpath)
        [x, y] = get_lasa_traj1(lasa_names[i])
        traj = np.hstack((x, y))
        print(np.shape(traj))
        my_mlfd = mlfd.SAMLfD()
        my_mlfd.add_traj(traj)
        my_mlfd.add_representation(ja.perform_ja_general, 'JA')
        my_mlfd.add_representation(lte.LTE_ND_any_constraints, 'LTE')
        my_mlfd.add_representation(dmp.perform_dmp_general, 'DMP')
        my_mlfd.add_metric(similaritymeasures.frechet_dist, is_dissim=True)
        my_mlfd.create_grid()
        my_mlfd.deform_traj()
        my_mlfd.calc_similarities()
        my_mlfd.save_to_h5(plt_fpath + 'mlfd_lasa_' + lasa_names[i] + '.h5')
        #my_mlfd.read_from_h5(plt_fpath + 'mlfd_lasa_' + lasa_names[i] + '.h5')
        my_mlfd.get_strongest_sims(0.1)
        my_mlfd.set_up_classifier()
        my_mlfd.plot_classifier_results(mode='save', filepath=plt_fpath)
        my_mlfd.plot_contour2D(mode='save', filepath=plt_fpath)
     
if __name__ == '__main__':
  main()