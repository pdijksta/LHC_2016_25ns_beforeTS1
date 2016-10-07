# This Script is intended to be used for PyEcloud benchmark simulations.
# The heat load on the arc is calculated using LHC data for a given fill and time.
# Comparing these to the heatloads from PyEcloud simulations will then allow for 
#   an estimation of the SEY parameter.
# It has been extended to also list the heat load on the Q6 quadrupoles.

# Written by Philipp Dijkstal, philipp.dijkstal@cern.ch

import sys
import os
import cPickle  # it is recommended to use cPickle over pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse as arg

import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.LHC_Energy as Energy
import LHCMeasurementTools.mystyle as ms
from LHCMeasurementTools.LHC_BCT import BCT
import LHCMeasurementTools.LHC_Heatloads as HL
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables


# CONFIG

default_avg_period = 0.1  # in hours
default_offset_period_begin = 0.1
default_offset_period_end = 0.35

first_correct_filln = 4474
colstr = {}
colstr[1] = 'b'
colstr[2] = 'r'
beams_list = [1, 2]
myfontsz = 16
ms.mystyle_arial(fontsz=myfontsz, dist_tick_lab=8)
pickle_name = 'heatload_arcs.pkl'


# PARSE ARGS

parser = arg.ArgumentParser(description='Calculate the heat loads on all arcs at a specified time for a given fill.' +
     'An average is taken. The heat loads should be stable at this point in time.')

parser.add_argument('fill', metavar='FILL', type=int, help='LHC fill number, must be at least %i.' % first_correct_filln)
parser.add_argument('time', metavar='TIME', type=str, help='Time after the specified FILL number has been set.\n Obtain it with the 016 script.')
parser.add_argument('-a', metavar='AVG_PERIOD', type=float, default=default_avg_period, 
                    help='The time in hours this program uses to find an average around the specified TIME.\nDefault: +/- %.2f' % default_avg_period)
parser.add_argument('-p', help='Save heat loads to pickle if entry does not exist.', action='store_true')
parser.add_argument('-f', help='Overwrite heat loads at the pickle if the entry does already exist.', action='store_true')
parser.add_argument('-n', help='No plot will be shown.', action='store_false')
parser.add_argument('-o', metavar=('T1', 'T2'), nargs=2, type=float, default=[default_offset_period_begin, default_offset_period_end], 
                    help='The time in hours this program uses to calculate an offset.\nDefault: %.2f - %.2f' % (default_offset_period_begin, default_offset_period_end))

args = parser.parse_args()
avg_period = args.a
filln = args.fill
time_of_interest_str = args.time
store_pickle = args.p or args.f
overwrite_pickle = args.f
show_plot = args.n
[offset_time_hrs_begin, offset_time_hrs_end] = args.o

time_of_interest = float(time_of_interest_str)
dict_main_key = str(filln) + str(time_of_interest)

# A minimum fill number is inferred from the 016 script
if filln < first_correct_filln:
    raise ValueError("Fill number too small. Look at the help for this script.")


# LOAD DATA

dict_hl_groups = {}
arc_keys_list = HL.variable_lists_heatloads['AVG_ARC']
quad_keys_list = HL.variable_lists_heatloads['Q6s_IR1'] \
        + HL.variable_lists_heatloads['Q6s_IR5'] \
        + HL.variable_lists_heatloads['Q6s_IR2'] \
        + HL.variable_lists_heatloads['Q6s_IR8'] 
model_key = 'LHC.QBS_CALCULATED_ARC.TOTAL'
impedance_keys = ['LHC.QBS_CALCULATED_ARC_IMPED.B1', 'LHC.QBS_CALCULATED_ARC_IMPED.B2']
synchRad_keys = ['LHC.QBS_CALCULATED_ARC_SYNCH_RAD.B1', 'LHC.QBS_CALCULATED_ARC_SYNCH_RAD.B2']
model_key_nice = 'Imp+SR'

all_keys = arc_keys_list + quad_keys_list + [model_key] + impedance_keys + synchRad_keys


with open('fills_and_bmodes.pkl', 'rb') as fid:
    dict_fill_bmodes = cPickle.load(fid)

fill_dict = {}
fill_dict.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv' % filln, verbose=False))
fill_dict.update(tm.parse_timber_file('fill_heatload_data_csvs/heatloads_fill_%d.csv' % filln, verbose=False))

bct_bx = {}
for beam_n in beams_list:
    bct_bx[beam_n] = BCT(fill_dict, beam=beam_n)
    # fbct not required

energy = Energy.energy(fill_dict, beam=1)
t_ref = dict_fill_bmodes[filln]['t_startfill']
tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_ref))

heatloads = SetOfHomogeneousNumericVariables(variable_list=all_keys, timber_variables=fill_dict)


# COMPUTE AVERAGES

# The heat loads calculated during this call of the script are stored here
this_hl_dict = {}

def get_output_key(input_key):
    """ Map keys for output dictionary. """
    if input_key in arc_keys_list:
        return input_key[0:3]
    elif input_key in quad_keys_list:
        return 'Q' + input_key[6:10]
    elif key == model_key:
        return model_key_nice
    elif key in impedance_keys:
        return 'Imp'
    elif key in synchRad_keys:
        return 'SR'
    else:
        raise ValueError('Wrong call for output_key: %s' % input_key)

def cut_arrays(arr, begin=(time_of_interest-avg_period), end=time_of_interest+avg_period):
    """
    Expects two column input: time, values.
    Returns the values for which time-avg_period < time < time+avg_period is satisfied
    """
    condition = np.logical_and(begin < arr[:,0], arr[:,0] < end)
    return np.extract(condition, arr[:,1])

def get_heat_loads(key):
    """
    Gets the heatloads of the specified key through the timber variables module.
    """
    time_heatload = np.array([(heatloads.timber_variables[key].t_stamps-t_ref)/3600., heatloads.timber_variables[key].values]).T
    time_heatload_cut = cut_arrays(time_heatload)
    avg_heatload = np.mean(time_heatload_cut)
    avg_heatload_sigma = np.std(time_heatload_cut)
    offset_cut = cut_arrays(time_heatload, begin=offset_time_hrs_begin, end=offset_time_hrs_end)
    avg_offset = np.mean(offset_cut)

    return [avg_heatload, avg_heatload_sigma, avg_offset]

def add_to_output_dict(input_key, avg_heatload, avg_heatload_sigma, offset):
    output_key = get_output_key(input_key)
    #print("The heatload %s is\n%.2f\t%.2f\twith offset %.2f\n" % (output_key, avg_heatload, avg_heatload_sigma,offset))
    this_hl_dict[output_key] = [avg_heatload, avg_heatload_sigma, offset]
    

for key in arc_keys_list + quad_keys_list + [model_key]:
    [hl, sigma, offset] = get_heat_loads(key)
    add_to_output_dict(key,hl,sigma,offset)

for key_list in [synchRad_keys, impedance_keys]:
    hl, sigma = 0, 0
    for key in key_list:
        [this_hl, this_sigma, offset] = get_heat_loads(key)
        hl += this_hl
        sigma += this_sigma
    add_to_output_dict(key,hl,sigma, 0)


# SAVE PICKLE

if store_pickle:
    print('Storing into pickle.')
    if not os.path.isfile(pickle_name):
        heatload_dict = {} 
    else :
        with open(pickle_name,'r') as hl_dict_file:
            heatload_dict = cPickle.load(hl_dict_file)
    
    filln_str = str(filln)
    t_o_i_str = str(time_of_interest)
    main_key = filln_str + ' ' + t_o_i_str
    
    if overwrite_pickle and main_key in heatload_dict.keys():
        print('Deleting old entry.')
        del heatload_dict[main_key]

    if main_key not in heatload_dict.keys():
        heatload_dict[main_key] = this_hl_dict
        with open(pickle_name, 'w') as hl_dict_file:
            cPickle.dump(heatload_dict,hl_dict_file)
    else:
        print('This entry already exists in the pickle, not storing!!\n')
    

# PLOTS

if not show_plot:
    sys.exit()

plt.close('all')
fig = plt.figure(1, figsize=(12, 10))

fig.patch.set_facecolor('w')
fig.canvas.set_window_title('LHC Arcs and Q6')
fig.set_size_inches(15., 8.)

plt.suptitle(' Fill. %d started on %s\nLHC Arcs and Q6' % (filln, tref_string))
plt.subplots_adjust(right=0.7, wspace=0.30)

sptotint = plt.subplot(3, 1, 1)
sphlcell = plt.subplot(3, 1, 2, sharex=sptotint)
sphlquad = plt.subplot(3, 1, 3, sharex=sptotint)
spenergy = sptotint.twinx()

# Energy
spenergy.plot((energy.t_stamps-t_ref)/3600., energy.energy/1e3, c='black', lw=2.)  # alpha=0.1)
spenergy.set_ylabel('Energy [TeV]')
spenergy.set_ylim(0, 7)

# Intensity
for beam_n in beams_list:
    sptotint.plot((bct_bx[beam_n].t_stamps-t_ref)/3600., bct_bx[beam_n].values, '-', color=colstr[beam_n])
    sptotint.set_ylabel('Total intensity [p+]')
    sptotint.grid('on')

# Heat Loads Arcs
for key in arc_keys_list + [model_key]:
    if key == model_key:
        label = 'Imp.+SR'
    else:
        label = key[0:3]  # Dictionary keys begin with S[0-9]{2}
    sphlcell.plot((heatloads.timber_variables[key].t_stamps-t_ref)/3600.,
                  heatloads.timber_variables[key].values, '--', lw=2., label=label)
    sphlcell.set_ylabel('Heatload [W]')

sphlcell.legend(prop={'size': myfontsz}, bbox_to_anchor=(1.1, 1),  loc='upper left')
sphlcell.grid('on')

# Heat Loads Quads
for key in quad_keys_list:
    label = 'Q' + key[6:10]
    sphlquad.plot((heatloads.timber_variables[key].t_stamps-t_ref)/3600.,
                  heatloads.timber_variables[key].values, lw=2., label=label)
sphlquad.legend(prop={'size': myfontsz}, bbox_to_anchor=(1.1, 1),  loc='upper left')
sphlquad.grid('on')
sphlquad.set_ylabel('Heatload [W]')

# Vertical line to indicate time_of_interest
for sp in [sphlcell, spenergy, sphlquad]:
    sp.axvline(time_of_interest, color='black')
    sp.axvline(offset_time_hrs_begin, color='black')
    sp.axvline(offset_time_hrs_end, color='black')
    for xx in [time_of_interest - avg_period, time_of_interest + avg_period]:
        sp.axvline(xx, ls='--', color='black')

plt.show()
