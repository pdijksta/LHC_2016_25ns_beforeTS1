import sys
import os
import cPickle as pickle

import LHCMeasurementTools.LHC_Fills as Fills
from LHCMeasurementTools.LHC_Fill_LDB_Query import save_variables_and_pickle
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables

fill_sublist = [5416]

variable_file = './GasFlowHLCalculator/variable_list.txt'
csv_folder = 'fill_all_heatload_data_csvs'
saved_pkl = csv_folder+'/saved_fills.pkl'
filepath =  csv_folder+'/all_heatload_data_fill'
fills_pkl_name = 'fills_and_bmodes.pkl'
fill_file = filepath + '_%i.csv'
pkl_file = filepath + '_%i.pkl'

fill_sublist_2 = []
for fill in fill_sublist:
    if os.path.isfile(pkl_file % fill):
        print('Pkl for fill %i already exists!' % fill)
    else:
        fill_sublist_2.append(fill)

with open(variable_file, 'r') as f:
    varlist = f.read().splitlines()[0].split(',')

if not os.path.isdir(csv_folder):
    os.mkdir(csv_folder)

with open(fills_pkl_name, 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)

save_variables_and_pickle(varlist=varlist, file_path_prefix=filepath, 
                          save_pkl=saved_pkl, fills_dict=dict_fill_bmodes, fill_sublist=fill_sublist_2)

for filln in fill_sublist_2:
    homogeneous_ob = SetOfHomogeneousNumericVariables(varlist, fill_file % filln)
    atd_ob = homogeneous_ob.aligned_object()
    with open(pkl_file % filln, 'w') as f:
        pickle.dump(atd_ob, f, protocol=-1)
    os.remove(fill_file % filln)
