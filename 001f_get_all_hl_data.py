import os
import cPickle as pickle

import LHCMeasurementTools.LHC_Fills as Fills
from LHCMeasurementTools.LHC_Fill_LDB_Query import save_variables_and_pickle
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables
import LHCMeasurementTools.myfilemanager as mfm

# Config
fill_sublist = [5222, 5223, 5416]
dt_seconds = 120

# File names
variable_files = ['./GasFlowHLCalculator/variable_list.txt', './GasFlowHLCalculator/missing_variables.txt']
h5_dir = '/eos/user/l/lhcscrub/timber_data_h5/cryo_heat_load_data/'
saved_pkl = h5_dir+'/saved_fills.pkl'
file_name = 'cryo_data_fill'
temp_filepath = './tmp/' + file_name
temp_file = temp_filepath + '_%i.csv'
h5_file = h5_dir + file_name + '_%i.h5'

fills_pkl_name = 'fills_and_bmodes.pkl'
with open(fills_pkl_name, 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)

fill_sublist_2 = []
for fill in fill_sublist:
    if os.path.isfile(h5_file % fill):
        print(h5_file % fill + ' already exists!')
    else:
        fill_sublist_2.append(fill)

varlist = []
for n in variable_files:
    with open(n, 'r') as f:
        varlist.extend(f.read().splitlines()[0].split(','))
for ii, var in enumerate(varlist):
    if '.POSST' not in var:
        varlist[ii] = var+'.POSST'

if not os.path.isdir(h5_dir):
    os.mkdir(h5_dir)

for filln in fill_sublist_2:
    print('Downloading csv for fill %i' % filln)
    save_variables_and_pickle(varlist=varlist, file_path_prefix=temp_filepath, save_pkl=saved_pkl, fills_dict=dict_fill_bmodes, fill_sublist=[filln])
    print('Aligning data for fill %i' % filln)
    htd_ob = SetOfHomogeneousNumericVariables(varlist, temp_file % filln).aligned_object(dt_seconds)
    print('Creating h5 file for fill %i' % filln)
    mfm.obj_to_h5(htd_ob, h5_file % filln)

    if os.path.isfile(h5_file % filln) and os.path.getsize(h5_file % filln) > 500e3:
        os.remove(temp_file % filln)
        print('Deleted temporary file %s!' % (temp_file % filln))
    else:
        print('Warning! Something went wrong for file %s!\nKeeping temporary file %s.' % (h5_file % filln, temp_file % filln))
