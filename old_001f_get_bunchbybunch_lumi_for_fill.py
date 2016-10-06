import LHCMeasurementTools.LHC_Fills as Fills
from LHCMeasurementTools.LHC_Fill_LDB_Query import save_variables_and_pickle

import pickle
import os

first_fill = 5000

csv_folder = 'fill_bunchbybunch_lumi_csvs'
filepath =  csv_folder+'/bunchbybunch_lumi_fill'

if not os.path.isdir(csv_folder):
    os.mkdir(csv_folder)
    
fills_pkl_name = 'fills_and_bmodes.pkl'
with open(fills_pkl_name, 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)

saved_pkl = csv_folder+'/saved_fills.pkl'

varlist = [
'ATLAS:BUNCH_LUMI_INST',
'CMS:BUNCH_LUMI_INST',
'ATLAS:LUMI_TOT_INST',
'CMS:LUMI_TOT_INST',
'ALICE:LUMI_TOT_INST',
'LHCB:LUMI_TOT_INST',
]

#remove fills in which we are not interested
filln_list = dict_fill_bmodes.keys()
for filln in filln_list:
    if filln<first_fill:
        del dict_fill_bmodes[filln]


save_variables_and_pickle(varlist=varlist, file_path_prefix=filepath, 
                          save_pkl=saved_pkl, fills_dict=dict_fill_bmodes)


