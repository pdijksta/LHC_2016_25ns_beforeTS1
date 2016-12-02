import h5py
import cPickle as pickle
import numpy as np
import LHCMeasurementTools.TimberManager as tm

save_dir = '/eos/user/l/lhcscrub/timber_data_h5/cryo_heat_load_data/'
pkl_file = './fill_all_heatload_data/all_heatload_data_fill_5219.pkl'


def save():

    with open(pkl_file, 'r') as f:
        data_0 = pickle.load(f)

    with h5py.File(save_dir + 'test.h5', 'w') as h5_file:
        h5_file.create_dataset('timestamps', data=data_0.timestamps)
        h5_file.create_dataset('data', data=data_0.data)
        h5_file.create_dataset('variables', data=data_0.variables)

def load():
    with h5py.File(save_dir + 'test.h5', 'r') as h5_file:
        print(h5_file.keys())
        print(h5_file['variables'][:5])

save()
load()
