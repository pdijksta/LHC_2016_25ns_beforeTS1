from __future__ import division
import sys
import os
import argparse
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt

import LHCMeasurementTools.myfilemanager as mfm
import LHCMeasurementTools.mystyle as ms
import LHCMeasurementTools.TimberManager as tm

import GasFlowHLCalculator.qbs_fill as qf
from GasFlowHLCalculator.data_QBS_LHC import arc_index, arc_list, Cell_list

plt.close('all')
ms.mystyle_arial(fontsz=16, dist_tick_lab=8)
colstr = {1: 'b', 2:'r'}
binwidth = 20
avg_pm_hrs = 0.1

fill_list = [5026, 5219, 5433]

def round_to(arr, precision):
    return np.round(arr/precision)*precision

with open('fills_and_bmodes.pkl', 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)

dict_dict = {}
for filln in fill_list:
    avg_time_hrs = (dict_fill_bmodes[filln]['t_start_STABLE'] - dict_fill_bmodes[filln]['t_startfill'])/3600.
    h5_file ='/eos/user/l/lhcscrub/timber_data_h5/cryo_heat_load_data/cryo_data_fill_%i.h5' % filln
    qbs_ob = qf.compute_qbs_fill(filln, use_dP=True)
    arc_hist_total, arc_hist_dict = qf.arc_histograms(qbs_ob, avg_time_hrs, avg_pm_hrs)
    dict_dict[filln] = {'total': arc_hist_total, 'arcs': arc_hist_dict, 'qbs': qbs_ob}

    tt = tm.UnixTimeStamp2UTCTimberTimeString(dict_fill_bmodes[filln]['t_startfill'])
    print('Fill %i: %s' % (filln, tt))

bins = np.arange(round_to(arc_hist_total.min(),binwidth)-binwidth, round_to(arc_hist_total.max(),binwidth)+binwidth*3/2, binwidth)

# Arcs
ls_list = ['-', '--', '-.', ':']

for arc_ctr, arc in enumerate(arc_list):
    arc_nr = arc[-2:]
    sp_ctr = arc_ctr % 4 + 1
    if sp_ctr == 1:
        fig = plt.figure()
        title = 'Arc heat load histograms'
        fig.canvas.set_window_title(title)
        fig.patch.set_facecolor('w')
    sp = plt.subplot(2,2,sp_ctr)
    sp.set_xlabel('Heat load [W]')
    sp.set_ylabel('# Half cells')
    sp.set_title('Arc %s' % arc_nr)
    sp.grid(True)

    for ls, filln in zip(ls_list,fill_list):
        arc_hist = dict_dict[filln]['arcs'][arc_nr]
        data = np.histogram(arc_hist, bins=bins)[0]
        sp.plot(bins[:-1], data, label='Fill %i' % filln, lw=2., drawstyle='steps', ls=ls)

    sp.set_ylim(None, sp.get_ylim()[1]+5)
    if sp_ctr == 2:
        sp.legend(bbox_to_anchor=(1.1,1))

# Full machine
fig = plt.figure()
title = 'Arc heat load histograms'
fig.canvas.set_window_title(title)
fig.patch.set_facecolor('w')
sp = plt.subplot(2,2,1)
sp.set_xlabel('Heat load [W]')
sp.set_ylabel('# Half cells')
sp.set_title('LHC')
sp.grid(True)
for ls, filln in zip(ls_list, fill_list):
    arc_hist_total = dict_dict[filln]['total']
    data = np.histogram(arc_hist_total, bins=bins)[0]
    sp.plot(bins[:-1], data, label='Fill %i' % filln, lw=2, ls=ls, drawstyle='steps')

sp.legend(bbox_to_anchor=(1.1,1))

# All cells
n_sps = 4
colors = ['red', 'green', 'blue']
for arc_ctr, arc in enumerate(arc_list):
    arc_nr = arc[-2:]
    sp_ctr = arc_ctr % n_sps + 1
    if sp_ctr == 1:
        fig = plt.figure()
        title = 'Cell heat loads'
        fig.canvas.set_window_title(title)
        fig.patch.set_facecolor('w')

    sp = plt.subplot(n_sps,1,sp_ctr)
    sp.set_title('Arc %s' % arc_nr)
    sp.set_ylabel('Heat load [W]')
    
    first, last = arc_index[arc_ctr,:]
    cell_list = []
    xx_list = []
    for color, (fill_ctr, filln) in zip(colors, enumerate(fill_list)):
        data = dict_dict[filln]['arcs'][arc_nr]
        for cell_ctr, cell_hl in enumerate(data):
            xx = cell_ctr*4 + fill_ctr
            if fill_ctr == 0:
                qbs_ctr = first + cell_ctr
                cell_list.append(Cell_list[qbs_ctr])
                xx_list.append(xx+1.5)
            if sp_ctr == 1 and cell_ctr == 0:
                label = filln
            else:
                label = None
            sp.bar(xx, cell_hl, color=color, label=label)
    
    sp.set_xticks(xx_list)
    sp.set_xticklabels(cell_list,fontsize=10)
    if sp_ctr == 1:
        sp.legend(bbox_to_anchor=(1.1,1))

# Delta hl vs hl
# LHC
fig = plt.figure()
title = 'Delta heat loads LHC June - October 2016'
plt.suptitle(title, fontsize=25)
fig.canvas.set_window_title(title)
fig.patch.set_facecolor('w')

sp = plt.subplot(2,2,1)
sp.set_xlabel('Heat load fill %i' % fill_list[0])
sp.set_ylabel('Reduction in heat load in fill %i' % fill_list[-1])
sp.grid(True)

hl_0 = dict_dict[fill_list[0]]['total']
hl_1 = dict_dict[fill_list[-1]]['total']
delta_hl = hl_0 - hl_1
sp.plot(hl_0, delta_hl,'.', markersize=7.)

# Arcs
sp = plt.subplot(2,2,2)
sp.set_xlabel('Heat load fill %i' % fill_list[0])
sp.set_ylabel('Reduction in heat load in fill %i' % fill_list[-1])
sp.grid(True)
for arc_ctr, arc in enumerate(arc_list):
    arc_nr = arc[-2:]
    data_0 = dict_dict[fill_list[0]]['arcs'][arc_nr]
    data_1 = dict_dict[fill_list[-1]]['arcs'][arc_nr]
    color = ms.colorprog(arc_ctr, arc_list)
    sp.plot(data_0, data_0-data_1,'.', color=color, label='Arc %s' % arc_nr, markersize=7.)
sp.legend(bbox_to_anchor=(1.1,1))

plt.show()
