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

import HeatLoadCalculators.impedance_heatload as ihl
import HeatLoadCalculators.synchrotron_radiation_heatload as srhl
import HeatLoadCalculators.FillCalculator as fc

import GasFlowHLCalculator.qbs_fill as qf

plt.close('all')
colstr = {1: 'b', 2:'r'}
binwidth = 20
offset_time = 't_start_INJPHYS'
offset_seconds = 600

parser = argparse.ArgumentParser()
parser.add_argument('fill', metavar='FILL', help='LHC fill number', type=int)
parser.add_argument('--nodp', help='Show recalculated without pressure drop', action='store_true')
parser.add_argument('--nolog', help='Do not show logged heat load', action='store_true')
parser.add_argument('--nohist', help='Do not show histograms.', action='store_true')
parser.add_argument('--noshow', help='Do not call plt.show.', action='store_true')
parser.add_argument('--pdsave', help='Save plots in pdijksta folder.', action='store_true')
parser.add_argument('--subtract-offset', help='Subtract HL offset.', action='store_true')
parser.add_argument('--selected', help='Only selected arcs/quads', action='store_true')
args = parser.parse_args()
filln = args.fill
no_use_dP = args.nodp
show_hist = not(args.nohist)

h5_file ='/eos/user/l/lhcscrub/timber_data_h5/cryo_heat_load_data/cryo_data_fill_%i.h5' % filln
if not os.path.isfile(h5_file):
    raise ValueError('%s does not exist' % h5_file)

arc_keys_list = HL.variable_lists_heatloads['AVG_ARC']
quad_keys = filter(lambda s: s.startswith('Q6s_'), HL.variable_lists_heatloads.keys())
quad_keys_list = []
for key in quad_keys:
    quad_keys_list.extend(HL.variable_lists_heatloads[key])

qbs_ob = qf.compute_qbs_fill(filln, use_dP=True)
qbs_arc_avg = qf.compute_qbs_arc_avg(qbs_ob).data
if no_use_dP:
    qbs_no = qf.compute_qbs_fill(filln, use_dP=False)
    qbs_arc_avg_no = qf.compute_qbs_arc_avg(qbs_no).data

t_ref = qbs_ob.timestamps[0]

fill_dict = {}
data_folder_list = ['./', '/afs/cern.ch/project/spsecloud/LHC_2015_PhysicsAfterTS2/' , '/afs/cern.ch/project/spsecloud/LHC_2015_IntRamp25ns/']

for data_folder in data_folder_list:
    try:
        fill_dict.update(tm.parse_timber_file(data_folder+'/fill_basic_data_csvs/basic_data_fill_%d.csv' % filln, verbose=False))
        with open(data_folder+'/fills_and_bmodes.pkl', 'rb') as fid:
            dict_fill_bmodes = pickle.load(fid)
        avg_time_hrs = (dict_fill_bmodes[filln]['t_start_STABLE'] - dict_fill_bmodes[filln]['t_startfill'])/3600.
        break
    except :
        try:
            fill_dict.update(tm.parse_timber_file(data_folder+'/fill_csvs/fill_%d.csv' % filln, verbose=False))
            with open(data_folder+'/fills_and_bmodes.pkl', 'rb') as fid:
                dict_fill_bmodes = pickle.load(fid)
            avg_time_hrs = (dict_fill_bmodes[filln]['t_start_STABLE'] - dict_fill_bmodes[filln]['t_startfill'])/3600.
            break
        except:
            pass
else:
    raise


try:
    fill_dict.update(tm.parse_timber_file('./fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_%d.csv' % filln, verbose=False))
except:
    pass

if not args.nolog:
    fill_dict.update(tm.parse_timber_file('./fill_heatload_data_csvs/heatloads_fill_%d.csv' % filln, verbose=False))
    heatloads = SetOfHomogeneousNumericVariables(variable_list=arc_keys_list+quad_keys_list, timber_variables=fill_dict)
energy = Energy.energy(fill_dict, beam=1)
bct_bx = {}
for beam_n in colstr:
    bct_bx[beam_n] = BCT(fill_dict, beam=beam_n)

hli_calculator  = ihl.HeatLoadCalculatorImpedanceLHCArc()
hlsr_calculator  = srhl.HeatLoadCalculatorSynchrotronRadiationLHCArc()

hl_imped_fill = fc.HeatLoad_calculated_fill(fill_dict, hli_calculator)
hl_sr_fill = fc.HeatLoad_calculated_fill(fill_dict, hlsr_calculator)

figs = []
ms.mystyle(12)
title = 'Recalculated arc heat loads %i' % filln
fig = ms.figure(title, figs)

# Arc half cell histograms
lhc_hist_dict = qf.lhc_histograms(qbs_ob, avg_time_hrs, 0.1)
arc_hist_dict = lhc_hist_dict['arcs']
arc_hist_total = lhc_hist_dict['total']

# Intensity and Energy
sptotint = plt.subplot(2, 2, 1)
sptotint.set_ylabel('Total intensity [p+]')
sptotint.grid('on')
for beam_n in colstr:
    sptotint.plot((bct_bx[beam_n].t_stamps-t_ref)/3600., bct_bx[beam_n].values, '-', color=colstr[beam_n], lw=2)

spenergy = sptotint.twinx()
spenergy.plot((energy.t_stamps-t_ref)/3600., energy.energy/1e3, c='black', lw=2.)  # alpha=0.1)
spenergy.set_ylabel('Energy [TeV]')
spenergy.set_ylim(0, 7)

# Heat loads arcs

t_start_injphys = (dict_fill_bmodes[filln]['t_start_INJPHYS']-t_ref)/3600.
def find_offset(tt, yy):
    mask = np.logical_and(tt < t_start_injphys, tt > t_start_injphys - 600/3600.)
    return np.mean(yy[mask])

arc_keys_list.sort()
sphlcell = plt.subplot(2,2,3, sharex=sptotint)
sphlcell.set_xlabel('Time [h]')

# separate figure
fig2 = ms.figure('Arcs only %i. Subtract offset %s' % (filln, args.subtract_offset), figs, figsize=(14,10))
fig2.subplots_adjust(wspace=0.7)
plt.suptitle('')
sp = plt.subplot(2,2,1)


fig2 = ms.figure('Quads only %i' % filln, figs, figsize=(14,10))
fig2.subplots_adjust(wspace=0.7)
sp2 = plt.subplot(2,2,2)
if args.selected:
    sp.set_ylim(45, 120)
    sp.set_xlim(2,4)
    sp2.set_ylim(10, 105)
    sp2.set_xlim(2,4)
for sp_ in (sp, sphlcell):
    sp_.set_title('Average arc half cells')
    sp_.grid(True)
    sp_.set_ylabel('Heat load [W]')

def plot_both(*args, **kwargs):
    sp.plot(*args, **kwargs)
    sphlcell.plot(*args, **kwargs)

tt = (qbs_ob.timestamps - t_ref)/3600.
for arc_ctr, key in enumerate(arc_keys_list):
    color = ms.colorprog(arc_ctr, len(arc_keys_list)+1)
    label = key[:3]
    if args.selected and label not in ('S12', 'S56'):
        continue
    # Logged
    # Recalculated
    if arc_ctr == 0:
        label1 = label+' with dP'
        label2 = label+' without dP'
    else:
        label1 = label
        label2 = None
    if args.subtract_offset:
        subtract = find_offset(tt, qbs_arc_avg[:,arc_ctr])
    else:
        subtract = 0

    plot_both(tt, qbs_arc_avg[:,arc_ctr]-subtract,'-', color=color, lw=2., label=label1)
    if no_use_dP:
        if args.subtract_offset:
            subtract = find_offset(tt, qbs_arc_avg_no[:,arc_ctr])
        else:
            subtract = 0
        plot_both(tt, qbs_arc_avg_no[:,arc_ctr]-subtract,'-.', color=color, lw=2., label=label2)
    if not args.nolog:
        xx_time = (heatloads.timber_variables[key].t_stamps-t_ref)/3600.
        yy_heatloads = (heatloads.timber_variables[key].values)
        if args.subtract_offset:
            subtract = find_offset(xx_time, yy_heatloads)
        else:
            subtract = 0
        if arc_ctr == 0:
            label3 = label +' logged'
        else:
            label3 = None
        plot_both(xx_time, yy_heatloads-subtract, '--', lw=2., label=label3, color=color)

if not args.selected:
    model_time = (hl_imped_fill.t_stamps - t_ref)/3600.
    hl_imp = hl_imped_fill.heat_load_calculated_total*53
    hl_sr = hl_sr_fill.heat_load_calculated_total*53
    plot_both(model_time, hl_imp, label='Imp', lw=2, color='0.5')
    plot_both(model_time, hl_sr+hl_imp, label='Imp+SR', lw=2, color='black')

    if args.subtract_offset:
        for sp_ in sp, sphlcell:
            for tt_ in t_start_injphys, (t_start_injphys-600/3600.):
                sp_.axvline(tt_, color='black', ls='--', lw=2)


plt.figure(fig.number)
# Heat load q6
quad_keys_list.sort()
sp_quad = plt.subplot(2,2,4, sharex=sptotint)
for sp_ in (sp_quad, sp2):
    sp_.set_title('Q6 standalones')
    sp_.grid(True)
    sp_.set_xlabel('Time [h]')
    sp_.set_ylabel('Heat load [W]')
fill_dict_recalc = qf.get_fill_dict(filln)
heatloads_recalc = SetOfHomogeneousNumericVariables(variable_list=quad_keys_list, timber_variables=fill_dict_recalc)

def plot_both(*args, **kwargs):
    sp2.plot(*args, **kwargs)
    sp_quad.plot(*args, **kwargs)

if no_use_dP:
    fill_dict_nodp = qf.get_fill_dict(filln, use_dP=False)
    heatloads_nodp = SetOfHomogeneousNumericVariables(variable_list=quad_keys_list, timber_variables=fill_dict_nodp)
for ctr, key in enumerate(quad_keys_list):
    color = ms.colorprog(ctr, len(quad_keys_list)+1)
    xx_time2 = (heatloads_recalc.timber_variables[key].t_stamps-t_ref)/3600.
    yy_heatloads2 = (heatloads_recalc.timber_variables[key].values)
    if no_use_dP:
        xx_time3 = (heatloads_nodp.timber_variables[key].t_stamps-t_ref)/3600.
        yy_heatloads3 = (heatloads_nodp.timber_variables[key].values)

    label = key[6:10]
    if args.selected and label not in ('06R2', '06L5'):
        print label
        continue
    # Recalculated
    if ctr == 0:
        label1, label2 = label +' with dP', label + ' without dP'
    else:
        label1, label2 = label, None
    plot_both(xx_time2, yy_heatloads2,'-', color=color, lw=2., label=label1)
    if no_use_dP:
        plot_both(xx_time3, yy_heatloads3,'-.', color=color, lw=2., label=label2)
    # Logged
    if not args.nolog:
        xx_time = (heatloads.timber_variables[key].t_stamps-t_ref)/3600.
        yy_heatloads = (heatloads.timber_variables[key].values)
        if ctr == 0:
            label3 = label+' logged'
        else:
            label3 = None
        plot_both(xx_time, yy_heatloads, '--', lw=2., label=label3, color=color)


tt_offset_1 = (dict_fill_bmodes[filln][offset_time] - dict_fill_bmodes[filln]['t_startfill'])/3600.
tt_offset_2 = tt_offset_1 - offset_seconds/3600.
tt_stable = (dict_fill_bmodes[filln]['t_start_STABLE'] - dict_fill_bmodes[filln]['t_startfill'])/3600.

if False:
    for sp_ in (sphlcell, sp, sp_quad, sp2):
        for tt, label in zip([tt_offset_1, tt_offset_2], ['Offset', None]):
            sp_.axvline(tt, ls='--', color='black', label=label)
        sp_.axvline(tt_stable, ls='--', color='red', label='Stable beams')

sphlcell.legend(bbox_to_anchor=(1.3,1))
sp.legend(bbox_to_anchor=(1,1), loc='upper left')
sp_quad.legend(bbox_to_anchor=(1.3,1))
sp2.legend(bbox_to_anchor=(1,1), loc='upper left')

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
    for fig in figs:
        fig.subplots_adjust(wspace=0.9)
        sf.pdijksta(fig)
if not args.noshow:
    plt.show()

