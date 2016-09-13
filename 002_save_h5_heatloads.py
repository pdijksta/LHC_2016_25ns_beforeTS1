import sys, os
BIN = os.path.expanduser("../")
sys.path.append(BIN)

import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.LHC_Energy as LHC_Energy
import LHCMeasurementTools.LHC_BCT as LHC_BCT
import LHCMeasurementTools.LHC_Heatloads as LHC_Heatloads

import h5py
import numpy as np
import pickle

h5folder = 'heatloads_fill_h5s'

fills_pkl_name = 'fills_and_bmodes.pkl'
with open(fills_pkl_name, 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)
    
if not os.path.isdir(h5folder):
    os.mkdir(h5folder)
    

for filln in sorted(dict_fill_bmodes.keys()):
    print 'Fill n.',filln
    h5filename = h5folder+'/heatloads_all_fill_%d.h5'%filln
    
    if dict_fill_bmodes[filln]['flag_complete'] is False:
        print "Fill incomplete --> no h5 convesion"
        continue    
    
    if os.path.isfile(h5filename) and dict_fill_bmodes[filln]['flag_complete'] is True:
        print "Already complete and in h5"
        continue

    try:
        dict_fill_data = {}
        dict_fill_data.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv'%filln, verbose=False))
        dict_fill_data.update(tm.parse_timber_file('fill_heatload_data_csvs/heatloads_fill_%d.csv'%filln, verbose=False))


        varlist = []

        varlist += LHC_BCT.variable_list()
        varlist += LHC_Energy.variable_list()
        for kk in LHC_Heatloads.variable_lists_heatloads.keys():
            varlist+=LHC_Heatloads.variable_lists_heatloads[kk]


        dict_to_h5 = {}

        for varname in varlist:
            #~ print varname
            dict_to_h5[varname+'!t_stamps'] = np.float_(dict_fill_data[varname].t_stamps)
            dict_to_h5[varname+'!values'] = dict_fill_data[varname].float_values()


        with h5py.File(h5filename, 'w') as fid:
            for kk in dict_to_h5.keys():
                fid[kk] = dict_to_h5[kk]
    except IOError as err:
        print 'Skipped!!! Got:'
        print err
