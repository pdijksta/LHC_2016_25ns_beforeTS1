from __future__ import division
import sys
import cPickle as pickle
import time
import re

import numpy as np
import matplotlib.pyplot as plt

import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.mystyle as ms
import LHCMeasurementTools.LHC_Heatloads as HL
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables
import LHCMeasurementTools.mystyle as ms
import LHCMeasurementTools.myfilemanager as mfm
import LHCMeasurementTools.LHC_Energy as Energy

from GasFlowHLCalculator.compute_QBS_LHC import compute_QBS_LHC
from GasFlowHLCalculator.data_QBS_LHC import arc_index, Cell_list

from m025c_use_hl_dict import main_dict as hl_dict

# Config
filln = 5219
avg_time_hrs = 2.
avg_pm_hrs = 0.1

myfontsz = 16
ms.mystyle_arial(fontsz=myfontsz, dist_tick_lab=8)
re_dev = re.compile('^QRLAA_(\d\d[RL]\d)_QBS\d{3}(_[QD]\d).POSST$')
plt.close('all')

# Definitions
keys = ['special_HC_Q1', 'special_HC_D2', 'special_HC_D3', 'special_HC_D4', 'special_total']
cells = ['13L5', '33L5', '13R4']
cells_and_new = cells + ['31L2']
affix_list = ['_Q1', '_D2', '_D3', '_D4']
beam_colors = {1: 'b', 2: 'r'}

h5_file ='/eos/user/l/lhcscrub/timber_data_h5/cryo_heat_load_data/cryo_data_fill_%i.h5' % filln

# Which cells are the special ones in the qbs data?
cell_index_dict = {}
for cell in cells_and_new:
    for index, var in enumerate(Cell_list):
        if cell in var:
            cell_index_dict[cell] = index
            break
    else:
        raise ValueError('Cell %s was not found' % cell)

variable_list = []
for key in keys:
    variable_list.extend(HL.variable_lists_heatloads[key])

with open('fills_and_bmodes.pkl', 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)

fill_dict = {}
fill_dict.update(tm.parse_timber_file('./fill_basic_data_csvs/basic_data_fill_%d.csv' % filln, verbose=False))
fill_dict.update(tm.parse_timber_file('./fill_heatload_data_csvs/heatloads_fill_%d.csv' % filln, verbose=False))

energy = Energy.energy(fill_dict, beam=1)
energy.t_stamps = (energy.t_stamps - energy.t_stamps[0])/3600.
t_ref = dict_fill_bmodes[filln]['t_startfill']
tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_ref))

heatloads = SetOfHomogeneousNumericVariables(variable_list=variable_list, timber_variables=fill_dict).aligned_object(dt_seconds=30)
y_min, y_max = -10, np.max(heatloads.data)+5
timestamps = (heatloads.timestamps - heatloads.timestamps[0])/3600.
mask_mean = np.abs(timestamps - avg_time_hrs) < avg_pm_hrs

atd_ob = mfm.h5_to_obj(h5_file)
QBS_ARC_AVG, arc_list, qbs, qbs_locals = compute_QBS_LHC(atd_ob, use_dP=False, return_qbs=True)
atd_tt = (atd_ob.timestamps - atd_ob.timestamps[0])/3600.
atd_mask_mean = np.abs(atd_tt - avg_time_hrs) < avg_pm_hrs
atd_mean_hl = []
for ctr in xrange(qbs.shape[1]):
    atd_mean_hl.append(np.mean(qbs[:,ctr][atd_mask_mean]))

s45_index = arc_list.index('ARC45')
first, last = arc_index[s45_index,:]
s45_qbs = qbs[:,first:last+1]
s45_mean_hl = []
for ctr in xrange(s45_qbs.shape[1]):
    s45_mean_hl.append(np.mean(s45_qbs[:,ctr][atd_mask_mean]))


# Plots
title = 'Special instrumented cells for fill %i' % filln
fig = plt.figure()
fig.canvas.set_window_title(title)
fig.patch.set_facecolor('w')
fig.subplots_adjust(left=.06, right=.88, top=.93, hspace=.38, wspace=.42)
plt.suptitle(title)
sp = None

# Cells
var_mean_dict = {}
for cell_ctr, cell in enumerate(cells):
    cell_vars = []
    for var in variable_list:
        if cell in var:
            cell_vars.append(var)
    sp_ctr = cell_ctr +1
    sp = plt.subplot(2,2,sp_ctr, sharex=sp)
    sp.set_title(cell)
    sp.grid('on')
    sp.set_xlabel('Time [h]')
    sp.set_ylabel('Heat load [W]')
    sp.set_ylim(y_min, y_max)
    sp2 = sp.twinx()
    sp2.set_ylabel('Energy [TeV]')
    sp2.plot(energy.t_stamps, energy.energy/1e3, c='black', lw=2.)
    
    summed = 0
    for ctr, var in enumerate(cell_vars):
        row_ctr = heatloads.variables.index(var)
        values = heatloads.data[:,row_ctr]
        mean_hl = np.mean(values[mask_mean])
        for affix in affix_list:
            if affix in var:
                label = affix[1:] 
                summed += values
                break
        else:
            label = 'Cell'
            var_mean_dict[cell] = mean_hl
        label += ' %.1f W' % mean_hl
        sp.plot(timestamps, values, label=label, ls='-', lw=2.)

    sp.axvline(avg_time_hrs, color='black')
    sp.plot(timestamps, summed, label='Sum', ls='-', lw=2.)
    sp.legend(bbox_to_anchor=(1.3,1), title='HL at %.1f h' % avg_time_hrs)

# Histogram
sp_hist = plt.subplot(2,2,4)
sp_hist.set_title('Heat loads at %.1f hours' % avg_time_hrs)
sp_hist.set_xlabel('Heat load [W]')
sp_hist.set_ylabel('# Half cells')
sp_hist.hist(atd_mean_hl, bins=np.arange(-50, 251, 10), label='All cells')
sp_hist.hist(s45_mean_hl, bins=np.arange(-50, 251, 10), label='S45')
sp_hist.axvline(np.mean(atd_mean_hl), color='blue', lw=2., label='Mean LHC')
sp_hist.axvline(np.mean(s45_mean_hl), color='green', lw=2., label='Mean S45')
colors = ['red', 'orange', 'brown', 'black']

mask_mean_qbs = np.abs(atd_tt - avg_time_hrs) < avg_pm_hrs
for ctr, cell in enumerate(cells_and_new):
    index = cell_index_dict[cell]
    mean = np.mean(qbs[:,index][mask_mean_qbs])
    sp_hist.axvline(mean, label=cell, color=colors[ctr], lw=2.)
sp_hist.legend(bbox_to_anchor=(1.3,1))


# Compare dipoles to quads
fig_dev = plt.figure()
fig_dev.canvas.set_window_title(title)
fig_dev.patch.set_facecolor('w')
fig_dev.subplots_adjust(left=.06, right=.84, top=.93, hspace=.38, wspace=.42)

sp_dip = plt.subplot(2,2,1)
sp_quad = plt.subplot(2,2,3, sharex=sp_dip)
dip_list = []
quad_list = []

ls_list = ['-', '--', '-.']

for var in variable_list:
    for affix in affix_list:
        if affix in var:
            if 'D' in affix:
                dip_list.append(var)
            elif 'Q' in affix:
                quad_list.append(var)

for sp, dev_list in zip((sp_dip, sp_quad), (dip_list, quad_list)):
    dev_list.sort()
    for ctr, dev in enumerate(dev_list):
        values = heatloads.data[:,heatloads.variables.index(dev)]
        info = re_dev.search(dev).groups()
        if info == ('33L5', '_D4') or info == ('33L5', '_D3'):
            continue
        color = ms.colorprog(affix_list.index(info[1]), affix_list)
        ls = ls_list[cells.index(info[0])]
        label = ''.join(info)
        sp.plot(timestamps, values, label=label, lw=2., color=color, ls=ls)
    sp.set_ylabel('Heat load [W]')
    sp.legend(bbox_to_anchor=(1.2,1))
    sp.set_ylim(-10,None)
    sp.set_xlabel('Time [h]')
    sp.grid('on')


# From large HL dict
sp = None
for cell_ctr, cell in enumerate(cells):
    sp_ctr = cell_ctr % 3 + 1
    if sp_ctr == 1:
        fig = plt.figure()
        fig.canvas.set_window_title('HL dict')
        fig.patch.set_facecolor('w')
        fig.subplots_adjust(left=.06, right=.84, top=.93, hspace=.38, wspace=.42)
    sp = plt.subplot(3,1,sp_ctr, sharex=sp)
    sp.set_xlabel('Fill number')
    sp.set_ylabel('Heat load [W]')
    sp.set_title(cell)
    sp.grid('on')
    for var in variable_list:
        if cell in var:
            info = re_dev.search(var)
            if info != None:
                name = ''.join(re_dev.search(var).groups())
                sp.plot(hl_dict['filln'], hl_dict['stable_beams']['heat_load'][name], lw=2., label=info.group(1))
    sp.legend(bbox_to_anchor=(1.05,1))
    sp.set_ylim(-10, 60)
    sp.set_xlim(4200, None)
    sp.grid('on')

plt.show()
