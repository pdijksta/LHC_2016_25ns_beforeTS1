from __future__ import division
import sys
import cPickle as pickle
import time
import re
import argparse

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

# Config
avg_pm_hrs = 0.1
fills_bmodes_name = './fills_and_bmodes.pkl'

parser = argparse.ArgumentParser()
parser.add_argument('filln', type=int)
parser.add_argument('-a', help='Point in time where to calculate the heat load', type=float, default=-1.)
parser.add_argument('-d', help='Plot for all fills in 2015/2016', action='store_true')
parser.add_argument('-w', help='Histogram bin width', default=20, type=float)
args = parser.parse_args()

filln = args.filln
avg_time_hrs = args.a
show_dict = args.d
binwidth = args.w

myfontsz = 16
ms.mystyle_arial(fontsz=myfontsz, dist_tick_lab=8)
re_dev = re.compile('^QRLAA_(\d\d[RL]\d)_QBS\d{3}(_[QD]\d).POSST$')
plt.close('all')

# Definitions
keys = ['special_HC_Q1', 'special_HC_D2', 'special_HC_D3', 'special_HC_D4', 'special_total']
cells = ['13L5', '33L5', '13R4']
new_cell = '31L2'
cells_and_new = cells + [new_cell]
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
if avg_time_hrs == -1.:
    avg_time_hrs = (dict_fill_bmodes[filln]['t_start_STABLE'] - dict_fill_bmodes[filln]['t_startfill'])/3600.

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

arc_cells_dict = {}
for ctr, arc_str in enumerate(arc_list):
    arc = arc_str[-2:]
    first, last = arc_index[ctr,:]
    arc_cells_dict[arc] = np.mean(qbs[atd_mask_mean,first:last+1], axis=0)
    if ctr == 0:
        arc_hist_total = arc_cells_dict[arc]
    else:
        arc_hist_total = np.append(arc_hist_total, arc_cells_dict[arc])

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
    if filln == 5277 and cell == '13L5':
        sp.set_ylim(-150, y_max)
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
    sp.plot(atd_tt, qbs[:,cell_index_dict[cell]], label='Recalc.', ls='-', lw=2., c='orange')
    sp.legend(bbox_to_anchor=(1.3,1), title='HL at %.1f h' % avg_time_hrs, fontsize=myfontsz)

# Histograms
def round_to(arr, precision):
    return np.round(arr/precision)*precision

# 1 for each arc
#bins = np.arange(round_to(arc_hist_total.min(),binwidth)-binwidth, round_to(arc_hist_total.max(),binwidth)+binwidth*3/2, binwidth)
bins = np.arange(-50, 251, 300./11.)
ctr = 0
for arc, data in arc_cells_dict.iteritems():
    sp_ctr = ctr % 4 + 1
    if sp_ctr == 1:
        fig = plt.figure()
        title = 'Fill %i: Heat loads at %.1f hours' % (filln, avg_time_hrs)
        fig.canvas.set_window_title(title)
        plt.suptitle(title)
        fig.patch.set_facecolor('w')
    sp = plt.subplot(2,2,sp_ctr)
    sp.hist(arc_hist_total, bins=bins, alpha=0.5, color='blue')
    sp.hist(data, bins=bins, color='green')
    sp.axvline(np.mean(data), lw=2., color='green')
    sp.axvline(np.mean(arc_hist_total), lw=2., color='blue')
    sp.grid('on')
    sp.set_xlabel('Heat load [W]')
    sp.set_ylabel('# Half cells')
    sp.set_title('Arc %s' % arc)

    if arc == '45':
        colors = ['red', 'orange', 'brown']
        for cell_ctr, cell in enumerate(cells):
            mean = np.mean(qbs[atd_mask_mean,cell_index_dict[cell]])
            sp.axvline(mean, label=cell, color=colors[cell_ctr])
            sp.legend(bbox_to_anchor=(1.2,1))
    elif arc == '12':
        mean = np.mean(qbs[atd_mask_mean,cell_index_dict[new_cell]])
        sp.axvline(mean, label=new_cell, color='red')
        sp.legend(bbox_to_anchor=(1.2,1))
    ctr += 1

# 1 plot for all sectors
fig = plt.figure()
fig.canvas.set_window_title(title)
fig.patch.set_facecolor('w')
plt.suptitle(title)
sp_hist = plt.subplot(2,2,1)
sp_hist.set_xlabel('Heat load [W]')
sp_hist.set_ylabel('# Half cells')
sp_hist.set_title('Bin width: %i W' % binwidth)
for ctr, (arc, data) in zip(xrange(len(arc_cells_dict)), arc_cells_dict.iteritems()):
    hist, null = np.histogram(data, bins=bins)
    sp_hist.plot(bins[:-1]+10, hist, label='Arc %s' % arc, color=ms.colorprog(ctr, arc_cells_dict), markersize=4, marker='o', lw=2)
#hist, null = np.histogram(arc_hist_total, bins=bins)
#sp_hist.plot(bins[:-1]+10, hist,'.', label='All', markersize=3.)
sp_hist.legend(bbox_to_anchor=(1.2,1))


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

for sp, dev_list, title in zip((sp_dip, sp_quad), (dip_list, quad_list), ('Dipoles', 'Quadrupoles')):
    dev_list.sort()
    for ctr, dev in enumerate(dev_list):
        values = heatloads.data[:,heatloads.variables.index(dev)]
        info = re_dev.search(dev).groups()
        if info == ('33L5', '_D4') or info == ('33L5', '_D3'):
            continue
        color = ms.colorprog(affix_list.index(info[1]), affix_list)
        ls = ls_list[cells.index(info[0])]
        label = ''.join(info)
        sp.plot(timestamps, values, label=label.replace('_',' '), lw=2., color=color, ls=ls)
    sp.set_title(title)
    sp.set_ylabel('Heat load [W]')
    sp.legend(bbox_to_anchor=(1.2,1))
    sp.set_ylim(-10,None)
    sp.set_xlabel('Time [h]')
    sp.grid('on')


# From large HL dict
if show_dict:
    from m025c_use_hl_dict import main_dict as hl_dict, mask_dict, main_dict_2016

    mask = hl_dict['stable_beams']['n_bunches']['b1'] > 1000
    ff_2016 = main_dict_2016['filln'][0]
    hl_dict = mask_dict(hl_dict, mask)

    int_norm_factor = hl_dict['stable_beams']['intensity']['total']
    ylims = [(-.1e-13,4.5e-13), (-10, 120), (-0.5,5), (-0.5,9)]
    norm_factors = [int_norm_factor, 1., 0., 0.]
    titles = ['Normalized by intensity', 'Heat loads', 'Normalized to 1', 'Normalized by intensity and to 1']
    units = ['[W/p]', '[W]', 'arb. units', 'arb. units']

    for ctr, norm_factor, ylim, title, unit in zip(xrange(len(norm_factors)), norm_factors, ylims, titles, units):
        sp = None
        for cell_ctr, cell in enumerate(cells):
            sp_ctr = cell_ctr % 3 + 1
            if sp_ctr == 1:
                fig = plt.figure()
                fig.canvas.set_window_title('HL dict' + title)
                fig.patch.set_facecolor('w')
                fig.subplots_adjust(left=.06, right=.84, top=.93, hspace=.38, wspace=.42)
                plt.suptitle(title, fontsize=25)
            sp = plt.subplot(3,1,sp_ctr, sharex=sp)
            sp.set_xlabel('Fill number')
            sp.set_ylabel('Heat load %s' % unit)
            sp.set_title(cell)
            sp.grid('on')
            names = [cell]
            labels = [cell]
            for var in variable_list:
                if cell in var:
                    info = re_dev.search(var)
                    if info != None:
                        names.append(''.join(re_dev.search(var).groups()))
                        labels.append(info.group(2).replace('_',' '))
                    else:
                        print('None: ', cell, var)

            for name, label in zip(names, labels):
                if ctr == 2:
                    norm_factor_2 = np.mean(hl_dict['stable_beams']['heat_load'][name][-20:])
                elif ctr == 3:
                    norm_factor_2 = np.mean(hl_dict['stable_beams']['heat_load'][name][-20:]/int_norm_factor[-20:])*int_norm_factor
                else:
                    norm_factor_2 = norm_factor
                sp.plot(hl_dict['filln'], hl_dict['stable_beams']['heat_load'][name]/norm_factor_2,'.', lw=2., label=label, markersize=12)

            sp.axvline(ff_2016, color='black', lw=2.)
            sp.legend(bbox_to_anchor=(1.05,1))
            sp.set_ylim(*ylim)
            sp.grid('on')

plt.show()
