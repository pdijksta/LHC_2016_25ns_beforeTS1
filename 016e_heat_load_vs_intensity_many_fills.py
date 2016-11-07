import sys, os
import pickle
import time
import pylab as pl
import numpy as np

import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.mystyle as ms
from LHCMeasurementTools.LHC_FBCT import FBCT
from LHCMeasurementTools.LHC_BCT import BCT
from LHCMeasurementTools.LHC_BQM import filled_buckets, blength
import LHCMeasurementTools.LHC_Heatloads as HL
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables

import HeatLoadCalculators.impedance_heatload as ihl
import HeatLoadCalculators.synchrotron_radiation_heatload as srhl 
import HeatLoadCalculators.FillCalculator as fc

## Choose fills

#reference fills
# filln_list = [5026, 5219, 5433]

# BCMS fills (latest part of the year)
filln_list = [5416, 5340, 5274]

#  fills (start of BCMS)
filln_list = [5076, 5069, 5071, 5073, 5076, 5080, 5083, 5091]

# test this script
#filln_list = [5219]

## Config
savefig = False

first_correct_filln = 4474
output_folder = 'plots'
beams_list = [1,2]
group_name = 'Arcs'
myfontsz = 16

##
len_cell = HL.magnet_length['AVG_ARC'][0]
dict_hl = HL.variable_lists_heatloads['AVG_ARC']
pl.close('all')
ms.mystyle_arial(fontsz=myfontsz, dist_tick_lab=8)

with open('fills_and_bmodes.pkl', 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)

fig_vs_int = pl.figure(100, figsize=(9, 6))
fig_vs_int.canvas.set_window_title('All sectors')
fig_vs_int.patch.set_facecolor('w')
spvsint = pl.subplot(111)

fig_blen_vs_int = pl.figure(200, figsize=(9, 6))
fig_blen_vs_int.canvas.set_window_title('Bunch length')
fig_blen_vs_int.patch.set_facecolor('w')
sp_blen_vs_int = pl.subplot(111)

for sector in HL.sector_list():
    fig = pl.figure(sector, figsize=(9, 6))
    fig.canvas.set_window_title('Sector %i' % sector)
    fig.patch.set_facecolor('w')
    sp = pl.subplot(111)

fills_string = ''
for i_fill, filln in enumerate(filln_list):
    
    fills_string += '_%d'%filln
    colorfill = ms.colorprog(i_prog=i_fill, Nplots=len(filln_list))
    
    fill_dict = {}
    fill_dict.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv'%filln, verbose=False))
    fill_dict.update(tm.parse_timber_file('fill_heatload_data_csvs/heatloads_fill_%d.csv'%filln, verbose=False))
    fill_dict.update(tm.parse_timber_file('fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_%d.csv'%filln, verbose=False))

    colstr = {}
    colstr[1] = 'b'
    colstr[2] = 'r'

    t_fill_st = dict_fill_bmodes[filln]['t_startfill']
    t_fill_end = dict_fill_bmodes[filln]['t_endfill']
    t_ref=t_fill_st
    tref_string=time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_ref))

    fbct_bx = {}
    bct_bx = {}
    blength_bx = {}

    for beam_n in beams_list:
        fbct_bx[beam_n] = FBCT(fill_dict, beam = beam_n)
        bct_bx[beam_n] = BCT(fill_dict, beam = beam_n)
        blength_bx[beam_n] = blength(fill_dict, beam = beam_n)
    beam_n=None

    # Calculate number of bunches
    filled_buckets_1 = filled_buckets(fill_dict, beam=1)
    filled_buckets_2 = filled_buckets(fill_dict, beam=2)
    n_bunches_1 = max(filled_buckets_1.Nbun)
    n_bunches_2 = max(filled_buckets_2.Nbun)
    n_bunches = max(n_bunches_1, n_bunches_2)
    if n_bunches_1 != n_bunches_2:
        print('Fill %i: N bunches for beam 1: %i, for beam 2: %i, choosing %i' % (filln, n_bunches_1, n_bunches_2, n_bunches))

    heatloads = SetOfHomogeneousNumericVariables(variable_list=dict_hl, timber_variables=fill_dict)

    # CORRECT ARC AVERAGES
    if filln < first_correct_filln:
        arc_correction_factor_list = HL.arc_average_correction_factors()
        hl_corr_factors = []
        for ii, varname in enumerate(dict_hl):
            hl_corr_factors.append(arc_correction_factor_list[ii])
        heatloads.correct_values(hl_corr_factors)

    # Compute impedance and SR
    hli_calculator  = ihl.HeatLoadCalculatorImpedanceLHCArc()
    hlsr_calculator  = srhl.HeatLoadCalculatorSynchrotronRadiationLHCArc()

    hl_imped_fill = fc.HeatLoad_calculated_fill(fill_dict, hli_calculator, fbct_dict=fbct_bx, bct_dict=bct_bx, blength_dict=blength_bx)
    hl_sr_fill = fc.HeatLoad_calculated_fill(fill_dict, hlsr_calculator, fbct_dict=fbct_bx, bct_dict=bct_bx, blength_dict=blength_bx)

    for ii, kk in enumerate(heatloads.variable_list):
        sector = int(kk[1:3])
        colorcurr = ms.colorprog(i_prog=ii, Nplots=len(heatloads.variable_list))

        hl_t_stamps = heatloads.timber_variables[kk].t_stamps
        hl_values = heatloads.timber_variables[kk].values
        bct_total = bct_bx[1].interp(hl_t_stamps)+bct_bx[2].interp(hl_t_stamps)

        mask_offset = np.logical_and(t_ref + 300 < hl_t_stamps, hl_t_stamps < t_ref + 3600)
        mask_offset = np.logical_and(mask_offset, bct_total < 1e-12)
        if sum(mask_offset) > 0:
            offset = np.mean(hl_values[mask_offset])
            # print(kk, offset)
        else:
            print('No offset for %s' % kk)
            offset = 0.

        mask_he = hl_t_stamps > dict_fill_bmodes[filln]['t_stop_SQUEEZE']
        subtract_imped = np.interp(hl_t_stamps[mask_he], hl_imped_fill.t_stamps, hl_imped_fill.heat_load_calculated_total*len_cell)
        subtract_SR = np.interp(hl_t_stamps[mask_he], hl_imped_fill.t_stamps, hl_sr_fill.heat_load_calculated_total*len_cell)

        mask_calc_avail = np.logical_and(subtract_imped>0.1, subtract_SR>0.1)
        ecloud_hl = hl_values[mask_he] - offset - subtract_imped - subtract_SR

        binten_hl = (bct_bx[1].interp(hl_t_stamps[mask_he])+bct_bx[2].interp(hl_t_stamps[mask_he]))/float(n_bunches)/2.
        if i_fill == 0:
            spvsint_label = 'S' + str(sector)
        else:
            spvsint_label = None

        spvsint.plot(binten_hl[mask_calc_avail], ecloud_hl[mask_calc_avail], '.', color=colorcurr, lw=2., label=spvsint_label)

        fig = pl.figure(sector)
        fig.canvas.set_window_title('Sector %i' % sector)
        sp = pl.subplot(111)
        sp.set_xlim(0.6e11, 1.3e11)
        sp.set_ylim(-5, 130)
        sp.grid('on')
        sp.set_xlabel('Bunch intensity [p+]')
        sp.set_ylabel('Heat load from e-cloud [W/hc]')
        
        fig.subplots_adjust(right=0.7, wspace=0.30, bottom=.12, top=.87)
        fig.suptitle('Sector %i' % sector)

        sp.plot(binten_hl[mask_calc_avail], ecloud_hl[mask_calc_avail], '.', color=colorfill, lw=2., label=filln)
        if i_fill == len(filln_list)-1:
            sp_label = 'Offset'
        else:
            sp_label = None
        sp.axhline(offset, color=colorfill, ls='--', lw=2., label=sp_label)
        sp.legend(prop={'size':myfontsz}, bbox_to_anchor=(1.1, 1),  loc='upper left')
        if savefig:
            fig.savefig(output_folder+'/hl_vs_int_S%d_fill%s' % (sector, fills_string), dpi=200)

    t_bl = blength_bx[1].t_stamps
    mask_bl_he = t_bl>dict_fill_bmodes[filln]['t_stop_SQUEEZE']

    binten_bl = (bct_bx[1].interp(t_bl[mask_bl_he])+bct_bx[2].interp(t_bl[mask_bl_he]))/float(n_bunches)/2.
    sp_blen_vs_int.plot(binten_bl, blength_bx[1].avblen[mask_bl_he]/1e-9, '.', color=colorfill, lw=2., label=filln)

spvsint.set_xlim(0.2e11, 1.3e11)
spvsint.set_ylim(0, 130)
spvsint.grid('on')
spvsint.legend(prop={'size':myfontsz}, bbox_to_anchor=(1.1, 1),  loc='upper left')
spvsint.set_xlabel('Bunch intensity [p+]')
spvsint.set_ylabel('Heat load from e-cloud [W/hc]')

sp_blen_vs_int.set_xlim(0.6e11, 1.3e11)
sp_blen_vs_int.set_ylim(0.7, 1.3)
sp_blen_vs_int.grid('on')
sp_blen_vs_int.legend(prop={'size':myfontsz}, bbox_to_anchor=(1.1, 1),  loc='upper left')
sp_blen_vs_int.set_xlabel('Bunch intensity [p+]')
sp_blen_vs_int.set_ylabel('Bunch length [ns]')

pl.subplots_adjust(right=0.7, wspace=0.30)

fig_vs_int.subplots_adjust(right=0.7, wspace=0.30, bottom=.12, top=.87)
fig_vs_int.suptitle(' Fill. %d started on %s\n%s'%(filln, tref_string, group_name))
if savefig:
    fig_vs_int.savefig(output_folder+'/hl_vs_int_fill%s'%(fills_string), dpi=200)

#~ fig_blen_vs_int.set_size_inches(15., 8.)
fig_blen_vs_int.subplots_adjust(right=0.7, wspace=0.30, bottom=.12, top=.87)
if savefig:
    fig_blen_vs_int.savefig(output_folder+'/blen_vs_int_fill%s'%(fills_string), dpi=200)

pl.show()
