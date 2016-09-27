import sys, os
import pickle
import time
import pylab as pl
import numpy as np

import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.LHC_Energy as Energy
import LHCMeasurementTools.mystyle as ms
from LHCMeasurementTools.LHC_FBCT import FBCT
from LHCMeasurementTools.LHC_BCT import BCT
from LHCMeasurementTools.LHC_BQM import filled_buckets, blength
import LHCMeasurementTools.LHC_Heatloads as HL
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables

import scipy.io as sio

flag_bunch_length = True
flag_average = True
flag_fbct = False
plot_model = True # C
t_zero = None

#filln_list = [5043, 5045, 5052, 5078]
filln_list = [5045]
n_bunches = 2076
filln_list = [5173]
n_bunches = 2076
filln_list = [5219, 5222, 5223]
n_bunches = 2076
#filln_list = [5181]
#n_bunches = 2076
# filln_list = [5198]
# n_bunches = 2172
# filln_list = [5199]
# n_bunches = 2220

#~ filln = 5038
#~ n_bunches = 2041





blacklist = [\
'QRLAA_33L5_QBS947_D4.POSST',
'QRLAA_13R4_QBS947_D2.POSST',
'QRLAA_33L5_QBS947_D3.POSST',
#'QRLEC_05L1_QBS947.POSST',
#'QRLEA_05L8_QBS947.POSST',
#'QRLEA_06L8_QBS947.POSST',
#'QRLEA_05R8_QBS947.POSST']
#'S78_QBS_AVG_ARC.POSST']
]

beams_list = [1,2]


arc_correction_factor_list = HL.arc_average_correction_factors()
first_correct_filln = 4474


myfontsz = 16
pl.close('all')
ms.mystyle_arial(fontsz=myfontsz, dist_tick_lab=8)


dict_hl_groups = {}

dict_hl_groups['Arcs'] = HL.variable_lists_heatloads['AVG_ARC']
# This script always works for the arcs!!!!



with open('fills_and_bmodes.pkl', 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)

fig_vs_int = pl.figure(100, figsize=(9, 6))
fig_vs_int.patch.set_facecolor('w')
spvsint = pl.subplot(111)

fig_blen_vs_int = pl.figure(200, figsize=(9, 6))
fig_blen_vs_int.patch.set_facecolor('w')
sp_blen_vs_int = pl.subplot(111)


fills_string = ''
for i_fill, filln in enumerate(filln_list):
    
    fills_string += '_%d'%filln
    fill_dict = {}
    fill_dict.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv'%filln, verbose=False))
    fill_dict.update(tm.parse_timber_file('fill_heatload_data_csvs/heatloads_fill_%d.csv'%filln, verbose=False))
    fill_dict.update(tm.parse_timber_file('fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_%d.csv'%filln, verbose=False))

    dict_beam = fill_dict
    dict_fbct = fill_dict


    colstr = {}
    colstr[1] = 'b'
    colstr[2] = 'r'

    



    energy = Energy.energy(fill_dict, beam=1)

    t_fill_st = dict_fill_bmodes[filln]['t_startfill']
    t_fill_end = dict_fill_bmodes[filln]['t_endfill']
    t_fill_len = t_fill_end - t_fill_st


    t_min = dict_fill_bmodes[filln]['t_startfill']-0*60.
    t_max = dict_fill_bmodes[filln]['t_endfill']+0*60.


    t_ref=t_fill_st
    tref_string=time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_ref))



    fbct_bx = {}
    bct_bx = {}
    blength_bx = {}

    for beam_n in beams_list:

        fbct_bx[beam_n] = FBCT(fill_dict, beam = beam_n)
        bct_bx[beam_n] = BCT(fill_dict, beam = beam_n)
        if flag_bunch_length: blength_bx[beam_n] = blength(fill_dict, beam = beam_n)

    dict_hl_data =  fill_dict

    group_names = dict_hl_groups.keys()

    N_figures = len(group_names)

    sp1 = None


    ii = 0

    fig_h = pl.figure(i_fill, figsize=(8, 6))
    fig_h.patch.set_facecolor('w')




    sptotint = pl.subplot(3,1,1, sharex=sp1)
    sp1 = sptotint
    if flag_bunch_length: spavbl = pl.subplot(3,1,3, sharex=sp1)
    sphlcell = pl.subplot(3,1,2, sharex=sp1)
    spenergy = sptotint.twinx()

    spenergy.plot((energy.t_stamps-t_ref)/3600., energy.energy/1e3, c='black', lw=2.)#, alpha=0.1)
    spenergy.set_ylabel('Energy [TeV]')
    spenergy.set_ylim(0,7)


    for beam_n in beams_list:

        if flag_fbct: sptotint.plot((fbct_bx[beam_n].t_stamps-t_ref)/3600., fbct_bx[beam_n].totint, '.--', color=colstr[beam_n])
        sptotint.plot((bct_bx[beam_n].t_stamps-t_ref)/3600., bct_bx[beam_n].values, '-', color=colstr[beam_n], lw=2.)
        sptotint.set_ylabel('Total intensity [p+]')
        sptotint.grid('on')

        if flag_bunch_length:
            spavbl.plot((blength_bx[beam_n].t_stamps-t_ref)/3600., blength_bx[beam_n].avblen/1e-9, '.-', color=colstr[beam_n])
            spavbl.set_ylabel('Bunch length [ns]')
            spavbl.set_ylim(0.8,1.8)
            spavbl.grid('on')
            spavbl.set_xlabel('Time [h]')

    group_name = group_names[ii]
    pl.suptitle(' Fill. %d started on %s\n%s'%(filln, tref_string, group_name))
    fig_h.canvas.set_window_title(group_name)

    hl_var_names = dict_hl_groups[group_name][:]
    hl_var_names_copy = dict_hl_groups[group_name][:]
    for varname in hl_var_names_copy:
        if varname in blacklist:
            hl_var_names.remove(varname)

    heatloads = SetOfHomogeneousNumericVariables(variable_list=hl_var_names, timber_variables=dict_hl_data)
    hl_model = SetOfHomogeneousNumericVariables(variable_list=HL.variable_lists_heatloads['MODEL'], timber_variables=fill_dict)

    # CORRECT ARC AVERAGES
    if group_name == 'Arcs' and filln < first_correct_filln:
        hl_corr_factors = []
        for ii, varname in enumerate(dict_hl_groups[group_name]):
            if varname not in blacklist:
                hl_corr_factors.append(arc_correction_factor_list[ii])
        heatloads.correct_values(hl_corr_factors)



    if flag_average: hl_ts_curr, hl_aver_curr  = heatloads.mean()
    for ii, kk in enumerate(heatloads.variable_list):
        colorcurr = ms.colorprog(i_prog=ii, Nplots=len(heatloads.variable_list))
        if t_zero is not None:
            offset = np.interp(t_ref+t_zero*3600, heatloads.timber_variables[kk].t_stamps, heatloads.timber_variables[kk].values)
        else:
            offset=0.

        label = ''
        for st in kk.split('.POSST')[0].split('_'):
            if 'QRL' in st or 'QBS' in st or 'AVG' in st or 'ARC' in st:
                pass
            else:
                label += st + ' '
        label = label[:-1]
        
        sphlcell.plot((heatloads.timber_variables[kk].t_stamps-t_ref)/3600, heatloads.timber_variables[kk].values-offset,
            '-', color=colorcurr, lw=2., label=label)#.split('_QBS')[0])
        
        t_hl = heatloads.timber_variables[kk].t_stamps
        mask_he = t_hl>dict_fill_bmodes[filln]['t_stop_SQUEEZE']
        print(mask_he, t_hl)
        subtract = np.interp(t_hl[mask_he], hl_model.timber_variables['LHC.QBS_CALCULATED_ARC.TOTAL'].t_stamps, hl_model.timber_variables['LHC.QBS_CALCULATED_ARC.TOTAL'].values)
        
        print(kk)
        if i_fill == 0:
            spvsint.plot(bct_bx[beam_n].interp(t_hl[mask_he])/n_bunches, heatloads.timber_variables[kk].values[mask_he]-offset-subtract,
                '.', color=colorcurr, lw=2., label=label)
        else:
            spvsint.plot(bct_bx[beam_n].interp(t_hl[mask_he])/n_bunches, heatloads.timber_variables[kk].values[mask_he]-offset-subtract,
                '.', color=colorcurr, lw=2.)
    
    t_bl = blength_bx[beam_n].t_stamps
    mask_bl_he = t_bl>dict_fill_bmodes[filln]['t_stop_SQUEEZE']
    
    sp_blen_vs_int.plot(bct_bx[beam_n].interp(t_bl[mask_bl_he])/n_bunches, blength_bx[beam_n].avblen[mask_bl_he]/1e-9,
            '.', lw=2., label=filln)

    if plot_model and group_name == 'Arcs':
        kk = 'LHC.QBS_CALCULATED_ARC.TOTAL'
        label='Imp.+SR'
        sphlcell.plot((hl_model.timber_variables[kk].t_stamps-t_ref)/3600., hl_model.timber_variables[kk].values,
            '--', color='grey', lw=2., label=label)

    if flag_average: 
        if t_zero is not None:
            offset = np.interp(t_ref+t_zero*3600, hl_ts_curr, hl_aver_curr)
        else:
            offset=0.
        sphlcell.plot((hl_ts_curr-t_ref)/3600., hl_aver_curr-offset, 'k', lw=2)
    sphlcell.set_ylabel('Heat load [W]')


#~ sphlcell.set_xlabel('Time [h]')
sphlcell.legend(prop={'size':myfontsz}, bbox_to_anchor=(1.1, 1),  loc='upper left')
sphlcell.grid('on')

spvsint.set_xlim(0, 1.3e11)
spvsint.set_ylim(0, 130)
spvsint.grid('on')
spvsint.legend(prop={'size':myfontsz}, bbox_to_anchor=(1.1, 1),  loc='upper left')
spvsint.set_xlabel('Bunch intensity [p+]')
spvsint.set_ylabel('Heat load from e-cloud [W/hc]')

sp_blen_vs_int.set_xlim(0.7e11, 1.3e11)
sp_blen_vs_int.set_ylim(0.7, 1.2)
sp_blen_vs_int.grid('on')
sp_blen_vs_int.legend(prop={'size':myfontsz}, bbox_to_anchor=(1.1, 1),  loc='upper left')
sp_blen_vs_int.set_xlabel('Bunch intensity [p+]')
sp_blen_vs_int.set_ylabel('Bunch length [ns]')

pl.subplots_adjust(right=0.7, wspace=0.30)
fig_h.set_size_inches(15., 8.)

#~ fig_vs_int.set_size_inches(15., 8.)
fig_vs_int.subplots_adjust(right=0.7, wspace=0.30, bottom=.12, top=.87)
fig_vs_int.suptitle(' Fill. %d started on %s\n%s'%(filln, tref_string, group_name))
fig_vs_int.savefig('hl_vs_int_fill%s'%(fills_string), dpi=200)

#~ fig_blen_vs_int.set_size_inches(15., 8.)
fig_blen_vs_int.subplots_adjust(right=0.7, wspace=0.30, bottom=.12, top=.87)
fig_blen_vs_int.savefig('blen_vs_int_fill%s'%(fills_string), dpi=200)

pl.show()
