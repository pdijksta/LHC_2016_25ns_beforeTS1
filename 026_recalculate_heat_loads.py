from __future__ import division
import sys
import os
import argparse
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt

import LHCMeasurementTools.myfilemanager as mfm
import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.mystyle as ms
import LHCMeasurementTools.LHC_Heatloads as HL
import LHCMeasurementTools.LHC_Energy as Energy
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables
from LHCMeasurementTools.LHC_BCT import BCT

import GasFlowHLCalculator.qbs_fill as qf

colstr = {1: 'b', 2:'r'}
binwidth = 20

def main(filln, no_use_dP, show_hist=True, show_plot=True):
    h5_file ='/eos/user/l/lhcscrub/timber_data_h5/cryo_heat_load_data/cryo_data_fill_%i.h5' % filln
    if not os.path.isfile(h5_file):
        raise ValueError('%s does not exist' % h5_file)

    arc_keys_list = HL.variable_lists_heatloads['AVG_ARC']

    qbs_ob = qf.compute_qbs_fill(filln, use_dP=True)
    qbs_arc_avg = qf.compute_qbs_arc_avg(qbs_ob)
    if no_use_dP:
        qbs_no = qf.compute_qbs_fill(filln, use_dP=False)
        qbs_arc_avg_no = qf.compute_qbs_arc_avg(qbs_no)

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
    ms.mystyle_arial()
    fig = plt.figure()
    title = 'Recalculated arc heat loads %i' % filln
    fig.canvas.set_window_title(title)
    fig.patch.set_facecolor('w')
    fig.set_size_inches(15., 8.)
    plt.suptitle(title)

    # Arc half cell histograms
    arc_hist_total, arc_hist_dict = qf.arc_histograms(qbs_ob, avg_time_hrs, avg_pm_hrs=0.1)

    if show_plot:
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
                    fig = plt.figure()
                    title = 'Fill %i: Heat loads at %.1f hours' % (filln, avg_time_hrs)
                    fig.canvas.set_window_title(title)
                    plt.suptitle(title)
                    fig.patch.set_facecolor('w')
                sp = plt.subplot(2,2,sp_ctr)
                sp.hist(arc_hist_total, bins=bins, alpha=0.5, color='blue', weights=1./len(arc_hist_total)*np.ones_like(arc_hist_total), label='LHC')
                sp.hist(data, bins=bins, color='green', alpha=0.5, weights=1./len(data)*np.ones_like(data), label='Arc')
                sp.axvline(np.mean(data), lw=2., color='green')
                sp.axvline(np.mean(arc_hist_total), lw=2., color='blue')
                sp.grid('on')
                sp.set_xlabel('Heat load [W]')
                sp.set_ylabel('# Half cells (normalized)')
                sp.set_title('Arc %s' % arc)
            if sp_ctr == 2:
                sp.legend(bbox_to_anchor=(1.2,1))

            # 1 plot for all sectors
            fig = plt.figure()
            fig.canvas.set_window_title(title)
            fig.patch.set_facecolor('w')
            plt.suptitle(title)
            sp_hist = plt.subplot(2,2,1)
            sp_hist.set_xlabel('Heat load [W]')
            sp_hist.set_ylabel('# Half cells')
            sp_hist.set_title('Bin width: %i W' % binwidth)
            for ctr, (arc, data) in zip(xrange(len(arc_hist_dict)), arc_hist_dict.iteritems()):
                hist, null = np.histogram(data, bins=bins)
                sp_hist.step(bins[:-1]+10, hist, label='Arc %s' % arc, color=ms.colorprog(ctr, arc_hist_dict), lw=2)

            sp_hist.legend(bbox_to_anchor=(1.2,1))

        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fill', metavar='FILL', help='LHC fill number', type=int)
    parser.add_argument('--nodp', help='Do not calculate pressure drop.', action='store_true')
    parser.add_argument('--nohist', help='Do not show histograms.', action='store_false')
    args = parser.parse_args()
    main(args.fill, args.nodp, show_hist=args.nohist)
