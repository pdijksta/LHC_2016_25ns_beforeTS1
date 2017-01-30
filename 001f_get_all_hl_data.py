import os
import cPickle as pickle
import argparse
import time

import LHCMeasurementTools.lhc_log_db_query as lldb
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables
import LHCMeasurementTools.myfilemanager as mfm

import GasFlowHLCalculator.h5_storage as h5_storage

# Config
dt_seconds = 60
max_fill_hrs = 35
blacklist = []
blacklist.append(4948) # 116 hour long fill, exceeds memory
blacklist.append(5488) # 40 hour long fill, also exceeds memory
parser = argparse.ArgumentParser()
parser.add_argument('-r', help='reversed', action='store_true')
parser.add_argument('year', help='2015 or 2016', type=int)

args = parser.parse_args()
year = args.year

# File names
h5_dir_0 = '/eos/user/l/lhcscrub/timber_data_h5/cryo_heat_load_data/'

# [For all cells, for 3 special cells]
variable_files = ['./GasFlowHLCalculator/variable_list_complete.txt', './GasFlowHLCalculator/variable_list_special.txt']
h5_dirs = [h5_dir_0, h5_dir_0 + 'special_cells/']
file_names = ['cryo_data_fill', 'special_data_fill']
temp_filepaths = ['./tmp/' + f for f in file_names]
temp_files = [t + '_%i.csv' for t in temp_filepaths]
data_file_funcs = [h5_storage.get_data_file, h5_storage.get_special_data_file]

if year == 2015:
    fills_pkl_name = '/afs/cern.ch/project/spsecloud/LHC_2015_PhysicsAfterTS2/fills_and_bmodes.pkl'
elif year == 2016:
    fills_pkl_name = 'fills_and_bmodes.pkl'
else:
    raise ValueError('Invalid year')

with open(fills_pkl_name, 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)

for variable_file, h5_dir, file_name, temp_filepath, temp_file, data_file_func in \
        zip(variable_files, h5_dirs, file_names, temp_filepaths, temp_files, data_file_funcs):

    fill_sublist = sorted(dict_fill_bmodes.keys(), reverse=args.r)
    fill_sublist_2 = []
    data_files = os.listdir(os.path.dirname(data_file_func(0)))
    for filln in fill_sublist:
        if os.path.basename(data_file_func(filln)) in data_files:
            pass
        elif filln in blacklist:
            print('Fill %i is blacklisted' % filln)
        else:
            t_start_fill = dict_fill_bmodes[filln]['t_startfill']
            t_end_fill   = dict_fill_bmodes[filln]['t_endfill']
            fill_hrs = (t_end_fill - t_start_fill)/3600.
            if fill_hrs < max_fill_hrs:
                fill_sublist_2.append(filln)
            else:
                print('Fill %i exceeds %i hours and is skipped' % (filln, max_fill_hrs))
    print('Processing %i fills!' % len(fill_sublist_2))
    time.sleep(5)

    with open(variable_file, 'r') as f:
        varlist = f.read().splitlines()[0].split(',')
    for ii, var in enumerate(varlist):
        if '.POSST' not in var:
            raise ValueError('%s does not have a .POSST' % var)

    if not os.path.isdir(h5_dir):
        os.mkdir(h5_dir)

    for filln in fill_sublist_2:
        h5_file = data_file_func(filln)
        this_temp_file = temp_file % filln
        print('Downloading csv for fill %i' % filln)
        t_start_fill = dict_fill_bmodes[filln]['t_startfill']
        t_end_fill   = dict_fill_bmodes[filln]['t_endfill']
        lldb.dbquery(varlist, t_start_fill, t_end_fill, this_temp_file)
        print('Aligning data for fill %i' % filln)
        htd_ob = SetOfHomogeneousNumericVariables(varlist, this_temp_file).aligned_object(dt_seconds)
        print('Creating h5 file for fill %i' % filln)
        mfm.aligned_obj_to_h5(htd_ob, h5_file)

        if os.path.isfile(h5_file) and os.path.getsize(h5_file) > 500:
            os.remove(this_temp_file)
            print('Deleted temporary file %s!' % (this_temp_file))
        else:
            print('Warning! Something went wrong for file %s!\nKeeping temporary file %s.' % (h5_file % filln, temp_file % filln))
