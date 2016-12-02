import sys
import os
import cPickle as pickle
import h5py

import LHCMeasurementTools.LHC_Fills as Fills
from LHCMeasurementTools.LHC_Fill_LDB_Query import save_variables_and_pickle
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables

fill_sublist = [5416]
attrs = ['timestamps', 'data', 'variables']

variable_file = './GasFlowHLCalculator/variable_list.txt'
h5_dir = '/eos/user/l/lhcscrub/timber_data_h5/cryo_heat_load_data/'
saved_pkl = h5_dir+'/saved_fills.pkl'
file_name = 'cryo_data_fill'
temp_filepath = './tmp/' + file_name
temp_file = temp_filepath + '_%i.csv'
h5_file = h5_dir + file_name + '_%i.h5'

fills_pkl_name = 'fills_and_bmodes.pkl'

fill_sublist_2 = []
for fill in fill_sublist:
    if os.path.isfile(h5_file % fill):
        print('Pkl for fill %i already exists!' % fill)
    else:
        fill_sublist_2.append(fill)

with open(variable_file, 'r') as f:
    varlist = f.read().splitlines()[0].split(',')

if not os.path.isdir(h5_dir):
    os.mkdir(h5_dir)

with open(fills_pkl_name, 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)

save_variables_and_pickle(varlist=varlist, file_path_prefix=temp_filepath,
                          save_pkl=saved_pkl, fills_dict=dict_fill_bmodes, fill_sublist=fill_sublist_2)

for filln in fill_sublist_2:
    htd_ob = SetOfHomogeneousNumericVariables(varlist, temp_file % filln).aligned_object()
    with open(h5_file % filln, 'w') as h5_handle:
        for attr in attrs:
            h5_handle.create_dataset(attr, data=getattr(htd_ob, attr))
    os.remove(temp_file % filln)
