from __future__ import division
import sys
import os
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt

import LHCMeasurementTools.myfilemanager as mfm
import LHCMeasurementTools.mystyle as ms
import LHCMeasurementTools.TimberManager as tm
from hl_dicts.LHC_Heat_load_dict import main_dict as hl_dict

import GasFlowHLCalculator.qbs_fill as qf
from GasFlowHLCalculator.data_QBS_LHC import arc_index, arc_list, Cell_list
from GasFlowHLCalculator.data_qbs import data_qbs, arc_index, arc_list
Cell_list = data_qbs.Cell_list

plt.close('all')
ms.mystyle()

# Config
moment = 'stable_beams'
compare_to_logged = True

fill_list = [5219, 5222, 5223]
intensity_list = []

# Obtain intensity from large hl dict
for fill in fill_list:
    index = list(hl_dict['filln']).index(fill)
    bint_1 = hl_dict[moment]['intensity']['b1'][index]/hl_dict[moment]['n_bunches']['b1'][index]
    bint_2 = hl_dict[moment]['intensity']['b2'][index]/hl_dict[moment]['n_bunches']['b2'][index]
    #print('Fill %i\t%e\t%e' % (fill, bint_1, bint_2))
    intensity_list.append(np.mean([bint_1, bint_2]))


hl_data = np.zeros((len(fill_list), 8))
hl_logged = np.zeros_like(hl_data)
hl_qbs = []


for fill_ctr, filln in enumerate(fill_list):
    index = list(hl_dict['filln']).index(filln)
    tt = hl_dict[moment]['t_stamps'][index]
    qbs_ob = qf.compute_qbs_fill(filln)
    qbs_arc_avg = qf.compute_qbs_arc_avg(qbs_ob)
    tt_index = np.argmin(np.abs(qbs_ob.timestamps - tt))
    hl_data[fill_ctr,:] = qbs_arc_avg[tt_index,:]
    hl_qbs.append(qbs_ob.data[tt_index,:])
    if compare_to_logged:
        for arc_ctr, arc in enumerate(arc_list):
            sector = 'S' + arc[-2:]
            hl_logged[fill_ctr, arc_ctr] = hl_dict[moment]['heat_load'][sector][index]
hl_qbs = np.array(hl_qbs)
mask_arc = np.array(data_qbs.Type_list) == 'ARC'
hl_qbs = hl_qbs[:,mask_arc]


title = 'Heat load and bunch intensity'
fig = plt.figure()
fig.canvas.set_window_title(title)
fig.patch.set_facecolor('w')
fig.subplots_adjust(left=.09, bottom=.07, right=.88, top=.92, wspace=.42, hspace=.19)

sp = plt.subplot(2,2,1)
sp.set_ylabel('Total heat load [W/hcell]')
sp.set_xlabel('Bunch intensity [p/bunch]')
sp.set_title('Arcs')
sp.grid(True)
xx_fit = np.linspace(0.4e11,1.3e11,num=5)
for arc_ctr, arc in enumerate(arc_list):
    color = ms.colorprog(arc_ctr, arc_list)
    label = 'S' + arc[-2:]
    sp.plot(intensity_list, hl_data[:,arc_ctr],'.', label=label, markersize=7, marker='o', color=color)
    fit = np.polyfit(intensity_list, hl_data[:,arc_ctr],1)
    sp.plot(xx_fit, np.poly1d(fit)(xx_fit), color=color, lw=3)
    if compare_to_logged:
        fit = np.polyfit(intensity_list, hl_logged[:,arc_ctr],1)
        sp.plot(intensity_list, hl_logged[:,arc_ctr],'.', label=label+' logged', markersize=7, marker='x', color=color)
        sp.plot(xx_fit, np.poly1d(fit)(xx_fit),'--', color=color)

sp.legend(bbox_to_anchor=(1.3,1))
sp.set_xlim(0,None)
sp.set_ylim(0,None)

sp = plt.subplot(2,2,3)
sp.set_ylabel('Total heat load [W/hcell]')
sp.set_xlabel('Bunch intensity [p/bunch]')
sp.set_title('Cells')
sp.grid(True)
for cell_ctr in xrange(len(hl_qbs[0,:])):
    fit = np.polyfit(intensity_list,hl_qbs[:,cell_ctr],1)
    sp.plot(xx_fit, np.poly1d(fit)(xx_fit), lw=1)
sp.set_xlim(0,None)
#sp.set_ylim(0,None)

# Special instrumented dipoles
good_cells = ['13L5', '13R4']
dip_list = ['D2', 'D3', 'D4']

sp = plt.subplot(2,2,2)
sp.set_title('Special cell dipoles')
sp.set_ylabel('Total heat load [W]')
sp.set_xlabel('Bunch intensity [p/bunch]')
xx_dict = {}
for filln in fill_list:
    qbs_ob = qf.special_qbs_fill(filln)
    index = list(hl_dict['filln']).index(filln)
    tt = hl_dict[moment]['t_stamps'][index]
    tt_index = np.argmin(np.abs(qbs_ob['timestamps'] - tt))
    for cell in good_cells:
        for dip in dip_list:
            label = cell+' '+dip
            if label not in xx_dict:
                xx_dict[label] = []
            xx_dict[label].append(qbs_ob[cell][dip][tt_index])


for ctr, (label, value) in enumerate(xx_dict.iteritems()):
    fit = np.polyfit(intensity_list, value,1)
    color = ms.colorprog(ctr, xx_dict)
    sp.plot(intensity_list, value, '.', label=label, markersize=7, color=color, marker='o')
    sp.plot(xx_fit, np.poly1d(fit)(xx_fit), '-', color=color, lw=3)
sp.legend(bbox_to_anchor=(1.3,1))


plt.show()
