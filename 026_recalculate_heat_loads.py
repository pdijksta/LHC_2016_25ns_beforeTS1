from __future__ import division
import os
import argparse
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt

import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.mystyle as ms
import LHCMeasurementTools.LHC_Heatloads as HL
import LHCMeasurementTools.LHC_Energy as Energy
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables
from LHCMeasurementTools.LHC_BCT import BCT
import LHCMeasurementTools.savefig as sf

import GasFlowHLCalculator.qbs_fill as qf

colstr = {1: 'b', 2:'r'}
binwidth = 20

parser = argparse.ArgumentParser()
parser.add_argument('fill', metavar='FILL', help='LHC fill number', type=int)
parser.add_argument('--nodp', help='Do not calculate pressure drop.', action='store_true')
parser.add_argument('--nohist', help='Do not show histograms.', action='store_true')
parser.add_argument('--pdsave', help='Save plots in pdijksta folder.', action='store_true')
args = parser.parse_args()
filln = args.fill
no_use_dP = args.nodp
show_hist = not(args.nohist)

h5_file ='/eos/user/l/lhcscrub/timber_data_h5/cryo_heat_load_data/cryo_data_fill_%i.h5' % filln
if not os.path.isfile(h5_file):
    raise ValueError('%s does not exist' % h5_file)

arc_keys_list = HL.variable_lists_heatloads['AVG_ARC']

qbs_ob = qf.compute_qbs_fill(filln, use_dP=True)
qbs_arc_avg = qf.compute_qbs_arc_avg(qbs_ob).data
if no_use_dP:
    qbs_no = qf.compute_qbs_fill(filln, use_dP=False)
    qbs_arc_avg_no = qf.compute_qbs_arc_avg(qbs_no).data

t_ref = qbs_ob.timestamps[0]
with open('fills_and_bmodes.pkl', 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)
avg_time_hrs = (dict_fill_bmodes[filln]['t_start_STABLE'] - dict_fill_bmodes[filln]['t_startfill'])/3600.

fill_dict = {}
fill_dict.update(tm.parse_timber_file('./fill_basic_data_csvs/basic_data_fill_%d.csv' % filln, verbose=False))
fill_dict.update(tm.parse_timber_file('./fill_heatload_data_csvs/heatloads_fill_%d.csv' % filln, verbose=False))
heatloads = SetOfHomogeneousNumericVariables(variable_list=arc_keys_list, timber_variables=fill_dict)
energy = Energy.energy(fill_dict, beam=1)
bct_bx = {}
for beam_n in colstr:
    bct_bx[beam_n] = BCT(fill_dict, beam=beam_n)

plt.close('all')
figs = []
ms.mystyle_arial()
title = 'Recalculated arc heat loads %i' % filln
fig = ms.figure(title, figs)

# Arc half cell histograms
lhc_hist_dict = qf.lhc_histograms(qbs_ob, avg_time_hrs, 0.1)
arc_hist_dict = lhc_hist_dict['arcs']
arc_hist_total = lhc_hist_dict['total']

# Intensity and Energy
sptotint = plt.subplot(2, 1, 1)
sptotint.set_ylabel('Total intensity [p+]')
sptotint.grid('on')
for beam_n in colstr:
    sptotint.plot((bct_bx[beam_n].t_stamps-t_ref)/3600., bct_bx[beam_n].values, '-', color=colstr[beam_n])

spenergy = sptotint.twinx()
spenergy.plot((energy.t_stamps-t_ref)/3600., energy.energy/1e3, c='black', lw=2.)  # alpha=0.1)
spenergy.set_ylabel('Energy [TeV]')
spenergy.set_ylim(0, 7)

# Heat loads arcs
arc_keys_list.sort()
sphlcell = plt.subplot(2,1,2, sharex=sptotint)
sphlcell.grid('on')
sphlcell.set_xlabel('Time [h]')
sphlcell.set_ylabel('Heat load [W]')

tt = (qbs_ob.timestamps - t_ref)/3600.
for arc_ctr, key in enumerate(arc_keys_list):
    color = ms.colorprog(arc_ctr, len(arc_keys_list)+1)

    # Logged
    xx_time = (heatloads.timber_variables[key].t_stamps-t_ref)/3600.
    yy_heatloads = (heatloads.timber_variables[key].values)
    label = key[:3]
    if arc_ctr == 0:
        label += ' logged'
    sphlcell.plot(xx_time, yy_heatloads, '-', lw=2., label=label, color=color)
    # Recalculated
    if arc_ctr == 0:
        label1, label2 = 'with dP', 'without dP'
    else:
        label1, label2 = None, None
    sphlcell.plot(tt, qbs_arc_avg[:,arc_ctr],'--', color=color, lw=2., label=label1)
    if no_use_dP:
        sphlcell.plot(tt, qbs_arc_avg_no[:,arc_ctr],'-.', color=color, lw=2., label=label2)

sphlcell.legend(bbox_to_anchor=(1.1,1))

# Histogram for arcs
if show_hist:
    def round_to(arr, precision):
        return np.round(arr/precision)*precision

    # 1 for each arc
    bins = np.arange(round_to(arc_hist_total.min(),binwidth)-binwidth, round_to(arc_hist_total.max(),binwidth)+binwidth*3/2, binwidth)
    for ctr, (arc, data) in enumerate(arc_hist_dict.iteritems()):
        sp_ctr = ctr % 4 + 1
        if sp_ctr == 1:
            title = 'Fill %i: Heat loads at %.1f hours' % (filln, avg_time_hrs)
            fig = ms.figure(title, figs)
        sp = plt.subplot(2,2,sp_ctr)
        sp.hist(data, bins=bins, color='green', alpha=0.5, weights=1./len(data)*np.ones_like(data), label=None)
        sp.axvline(np.mean(data), lw=2., color='green', label='Mean Arc')
        sp.axvline(np.mean(arc_hist_total), lw=2., color='blue', label='Mean LHC')
        sp.grid('on')
        sp.set_xlabel('Heat load [W]')
        sp.set_ylabel('# Half cells (normalized)')
        sp.set_title('Arc %s' % arc)
        if sp_ctr == 2:
            sp.legend(bbox_to_anchor=(1.2,1))

    # 1 plot for all sectors
    title = 'Fill %i at %.1f h: LHC Arcs histograms' % (filln, avg_time_hrs)
    fig = ms.figure(title, figs)
    sp_hist = plt.subplot(2,2,1)
    sp_hist.set_xlabel('Heat load [W]')
    sp_hist.set_ylabel('# Half cells')
    sp_hist.set_title(title)
    sp_hist.grid(True)
    for ctr, arc in enumerate(sorted(arc_hist_dict.keys())):
        data = arc_hist_dict[arc]
        hist, null = np.histogram(data, bins=bins+ctr)
        sp_hist.step(bins[:-1]+10+ctr, hist, label='Arc %s' % arc, color=ms.colorprog(ctr, arc_hist_dict), lw=2)
    ymin, ymax = sp_hist.get_ylim()
    sp_hist.set_ylim(ymin, round_to(ymax,5)+5)
    sp_hist.legend(bbox_to_anchor=(1.2,1))

if args.pdsave:
    print('There are %i figs' % len(figs))
    for fig in figs:
        sf.pdijksta(fig)
        plt.close(fig)
else:
    plt.show()

