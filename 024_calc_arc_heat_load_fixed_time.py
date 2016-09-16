#!/usr/bin/python2

# This Script is intended to be used for PyEcloud benchmark simulations.
# The heat load on the arc is calculated using LHC data for a given fill and time.
# Comparing these to the heatloads from PyEcloud simulations will then allow for 
#   an estimation of the SEY parameter.

# Written by Philipp Dijkstal, 15.09.2016

import sys
import pickle
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
first_correct_filln = 4474
colstr = {}
colstr[1] = 'b'
colstr[2] = 'r'
beams_list = [1, 2]
myfontsz = 16
ms.mystyle_arial(fontsz=myfontsz, dist_tick_lab=8)


# PARSE ARGS

parser = arg.ArgumentParser(description='Calculate the heat loads on all arcs at a specified time for a given fill.' +
     'An average is taken. The heat loads should be stable at this point in time.')

# Fill number
parser.add_argument('fill', metavar='FILL', type=int, help='LHC fill number, must be at least %i.' % first_correct_filln)

# Point in Time
parser.add_argument('time', metavar='TIME', type=float, help='Time after the specified FILL number has been set.\n Obtain it with the 016 script.')

# Averaging period (optional)
parser.add_argument('-a', metavar='AVG_PERIOD', type=float, default=default_avg_period, 
                    help='The time in hours this program uses to find an average around the specified TIME.\nDefault: %.2f' % default_avg_period)

# Plot (optional)
parser.add_argument('-n', help='No plot will be shown', action='store_false')

args = parser.parse_args()
avg_period = args.a
filln = args.fill
time_of_interest = args.time
show_plot = args.n

if filln < first_correct_filln:
    print("Fill number too small. Look at the help for this function.")
    sys.exit()


# LOAD DATA

dict_hl_groups = {}
dict_hl_groups['Arcs'] = HL.variable_lists_heatloads['AVG_ARC']
group_names = dict_hl_groups.keys()
arc_keys_list = dict_hl_groups['Arcs']
model_key = 'LHC.QBS_CALCULATED_ARC.TOTAL'

with open('fills_and_bmodes.pkl', 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)

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

heatloads = SetOfHomogeneousNumericVariables(variable_list=arc_keys_list+[model_key], timber_variables=fill_dict)


# COMPUTE AVERAGES

def cut_arrays(arr):
    condition = np.logical_and(time_of_interest - avg_period < arr[:,0], arr[:,0] < time_of_interest + avg_period)
    return np.extract(condition, arr[:,1])

model_time_heatload = np.array([(heatloads.timber_variables[model_key].t_stamps-t_ref)/3600., heatloads.timber_variables[model_key].values]).T
model_time_heatload_cut = cut_arrays(model_time_heatload)

model_heatload = np.mean(model_time_heatload_cut)
model_headload_sigma = np.std(model_time_heatload_cut)

print("The heatload from impedance / SR is\n%.2f\t%.2f\n" % (model_heatload, model_headload_sigma))

for key in arc_keys_list:
    arc_time_heatload = np.array([(heatloads.timber_variables[key].t_stamps-t_ref)/3600., heatloads.timber_variables[key].values]).T
    arc_time_heatload_cut = cut_arrays(arc_time_heatload)

    avg_heatload = np.mean(arc_time_heatload_cut) - model_heatload
    avg_heatload_sigma = np.sqrt(np.var(arc_time_heatload_cut)+ model_headload_sigma**2)
    label = key[0:3]
    print("The heatload on arc %s attributed to e-cloud is\n%.2f\t%.2f" % (label, avg_heatload, avg_heatload_sigma))


# PLOTS

if not show_plot:
    sys.exit()

plt.close('all')
fig = plt.figure(1, figsize=(12, 10))

fig.patch.set_facecolor('w')
fig.canvas.set_window_title('LHC Arcs')
fig.set_size_inches(15., 8.)

plt.suptitle(' Fill. %d started on %s\nLHC Arcs' % (filln, tref_string))
plt.subplots_adjust(right=0.7, wspace=0.30)

sptotint = plt.subplot(2, 1, 1)
sphlcell = plt.subplot(2, 1, 2, sharex=sptotint)
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

# Heat Loads
for key in arc_keys_list + [model_key]:
    if key == model_key:
        label = 'Imp.+SR'
    else:
        label = key[0:2]  # Dictionary keys begin with S[0-9]{2}
    sphlcell.plot((heatloads.timber_variables[key].t_stamps-t_ref)/3600.,
                  heatloads.timber_variables[key].values, '--', lw=2., label=label)

sphlcell.legend(prop={'size': myfontsz}, bbox_to_anchor=(1.1, 1),  loc='upper left')
sphlcell.grid('on')

# Vertical line to indicate time_of_interest
for sp in [sphlcell, spenergy]:
    sp.axvline(time_of_interest, color='black')
    for xx in [time_of_interest - avg_period, time_of_interest + avg_period]:
        sp.axvline(xx, ls='--', color='black')

plt.show()
