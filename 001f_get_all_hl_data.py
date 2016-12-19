import os
import cPickle as pickle
import sys

import LHCMeasurementTools.LHC_Fills as Fills
from LHCMeasurementTools.LHC_Fill_LDB_Query import save_variables_and_pickle
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables
import LHCMeasurementTools.myfilemanager as mfm

# Config
dt_seconds = 60
max_fill_hrs = 35
blacklist = []
blacklist.append(4948) # 116 hour long fill, exceeds memory
blacklist.append(5488) # 40 hour long fill, also exceeds memory
year = 2015

# File names
variable_files = ['./GasFlowHLCalculator/variable_list.txt', './GasFlowHLCalculator/missing_variables.txt']
h5_dir = '/eos/user/l/lhcscrub/timber_data_h5/cryo_heat_load_data/'
saved_pkl = h5_dir+'/saved_fills.pkl'
file_name = 'cryo_data_fill'
temp_filepath = './tmp/' + file_name
temp_file = temp_filepath + '_%i.csv'
h5_file = h5_dir + file_name + '_%i.h5'

if year == 2015:
    fills_pkl_name = '/afs/cern.ch/project/spsecloud/LHC_2015_PhysicsAfterTS2/fills_and_bmodes.pkl'
elif year == 2016:
    fills_pkl_name = 'fills_and_bmodes.pkl'
else:
    sys.exit()

with open(fills_pkl_name, 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)

if os.path.isfile(saved_pkl):
    with open(saved_pkl, 'r') as f:
        saved_dict = pickle.load(f)
else:
    saved_dict = {}

fill_sublist = dict_fill_bmodes.keys()

fill_sublist_2 = []
for fill in fill_sublist:
    if fill in saved_dict or os.path.isfile(h5_file % fill):
        pass
    elif fill in blacklist:
        print('Fill %i is blacklisted' % fill)
    else:
        fill_hrs = (dict_fill_bmodes[fill]['t_endfill'] - dict_fill_bmodes[fill]['t_startfill'])/3600.
        if fill_hrs < max_fill_hrs:
            fill_sublist_2.append(fill)
        else:
            print('Fill %i exceeds %i hours and is skipped' % (fill, max_fill_hrs))
print('Processing %i fills!' % len(fill_sublist_2))

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

    if os.path.isfile(h5_file % filln) and os.path.getsize(h5_file % filln) > 500:
        os.remove(temp_file % filln)
        print('Deleted temporary file %s!' % (temp_file % filln))
    else:
        print('Warning! Something went wrong for file %s!\nKeeping temporary file %s.' % (h5_file % filln, temp_file % filln))
