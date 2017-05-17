from __future__ import division
import cPickle as pickle
import time
import pylab as pl
import numpy as np
import argparse

import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.LHC_Energy as Energy
import LHCMeasurementTools.mystyle as ms
import LHCMeasurementTools.savefig as sf
from LHCMeasurementTools.LHC_FBCT import FBCT
from LHCMeasurementTools.LHC_BCT import BCT
from LHCMeasurementTools.LHC_BQM import blength
import LHCMeasurementTools.LHC_Heatloads as HL
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables

import HeatLoadCalculators.impedance_heatload as ihl
import HeatLoadCalculators.synchrotron_radiation_heatload as srhl
import HeatLoadCalculators.FillCalculator as fc
import GasFlowHLCalculator.qbs_fill as qf

parser = argparse.ArgumentParser()
parser.add_argument('filln_list', metavar='Fill', type=int, help='LHC fill numbers.', nargs='+')
parser.add_argument('--no-bunch-length', help='Do not plot bunch length against intensity.', action='store_true')
parser.add_argument('--average', help='Plot average LHC heat loads.', action='store_true')
parser.add_argument('--fbct', help='Plot FBCT intensity.', action='store_true')
parser.add_argument('--no-model', help='Do not plot model heat load.', action='store_true')
parser.add_argument('--t0', metavar='T_0', help='Compute offset at t_0.', type=float)
parser.add_argument('--use-recalc', help='Use logged instead of recalculated data.', action='store_true')
parser.add_argument('--savefig', help='Save figures in pdijksta dir.', action='store_true')
parser.add_argument('--make-table-bint', metavar='B_int [e11]', help='Print heat loads at a certain bunch intensity.', type=float)
parser.add_argument('--make-table-blen', metavar='Bunch length [ns]', help='Print heat loads at a certain bunch length.', type=float)
parser.add_argument('--make-table-time', metavar='Time [h]', help='Print heat loads at a certain moment', type=float)
args = parser.parse_args()


filln_list = args.filln_list
flag_bunch_length = not args.no_bunch_length
flag_average = args.average
flag_fbct = args.fbct
plot_model = not args.no_model
t_zero = args.t0
use_recalculated = args.use_recalc
savefig = args.savefig
make_table = args.make_table_bint or args.make_table_blen or args.make_table_time

if t_zero:
    print('Heat load offsets are subtracted at t_0=%.2f' % t_zero)
else:
    print('Heat load offsets are not subtracted!')

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
fill_arc_hl_dict = {}

arc_correction_factor_list = HL.arc_average_correction_factors()
first_correct_filln = 4474

myfontsz = 14
pl.close('all')
ms.mystyle_arial(fontsz=myfontsz, dist_tick_lab=8)

dict_hl_groups = {}
dict_hl_groups['Arcs'] = HL.variable_lists_heatloads['AVG_ARC']

colstr = {1: 'b', 2: 'r'}

with open('fills_and_bmodes.pkl', 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)

fig_vs_int = pl.figure(100, figsize=(12, 10))
fig_vs_int.patch.set_facecolor('w')
#~ fig_vs_int.set_size_inches(15., 8.)
#fig_vs_int.suptitle(' Fill. %d started on %s\n%s'%(filln, tref_string, group_name))
fig_vs_int.subplots_adjust(right=0.88, wspace=0.72, bottom=.12, top=.95, left=0.06)

spvsint = pl.subplot(2,2,1)
spvsint.grid('on')
spvsint.set_xlabel('Bunch intensity [p+]')
spvsint.set_ylabel('Heat load from e-cloud [W/hc]')

#fig_blen_vs_int = pl.figure(200, figsize=(9, 6))
#fig_blen_vs_int.patch.set_facecolor('w')
sp_blen_vs_int = pl.subplot(2,2,2)
sp_blen_vs_int.grid('on')
sp_blen_vs_int.set_xlabel('Bunch intensity [p+]')
sp_blen_vs_int.set_ylabel('Bunch length [ns]')

hli_calculator  = ihl.HeatLoadCalculatorImpedanceLHCArc()
hlsr_calculator  = srhl.HeatLoadCalculatorSynchrotronRadiationLHCArc()

fills_string = ''
for i_fill, filln in enumerate(filln_list):
    fills_string += '_%d'%filln
    fill_dict = {}
    fill_dict.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv'%filln, verbose=False))
    fill_dict.update(tm.parse_timber_file('fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_%d.csv'%filln, verbose=False))
    if use_recalculated:
        fill_dict.update(qf.get_fill_dict(filln))
    else:
        fill_dict.update(tm.parse_timber_file('fill_heatload_data_csvs/heatloads_fill_%d.csv'%filln, verbose=False))

    energy = Energy.energy(fill_dict, beam=1)

    t_ref = dict_fill_bmodes[filln]['t_startfill']
    tref_string=time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_ref))

    fbct_bx = {}
    bct_bx = {}
    blength_bx = {}
    n_bunches_bx = {}

    for beam_n in (1,2):
        fbct_bx[beam_n] = FBCT(fill_dict, beam = beam_n)
        bint = fbct_bx[beam_n].bint
        min_int = 0.1 * np.max(bint)
        mask_filled = bint > min_int
        n_bunches_bx[beam_n] = np.max(np.sum(mask_filled, axis=1))
        bct_bx[beam_n] = BCT(fill_dict, beam = beam_n)
        if flag_bunch_length:
            blength_bx[beam_n] = blength(fill_dict, beam = beam_n)

    print(filln, 'Number of bunches:', n_bunches_bx)
    if n_bunches_bx[1] != n_bunches_bx[2]:
        print('Not the same number of bunches! Choosing beam 1.')
    n_bunches = n_bunches_bx[1]

    fig_h = pl.figure(i_fill, figsize=(8, 6))
    fig_h.patch.set_facecolor('w')
    fig_h.subplots_adjust(right=0.7, wspace=0.30)

    sptotint = pl.subplot(3,1,1)
    sptotint.grid(True)
    sptotint.set_ylabel('Total intensity [p+]')
    spenergy = sptotint.twinx()
    spenergy.set_ylabel('Energy [TeV]')
    spenergy.set_ylim(0,7)
    sp1 = sptotint

    sphlcell = pl.subplot(3,1,2, sharex=sp1)
    sphlcell.set_ylabel('Heat load [W]')
    sphlcell.grid(True)

    if flag_bunch_length:
        spavbl = pl.subplot(3,1,3, sharex=sp1)
        spavbl.set_ylabel('Bunch length [ns]')
        spavbl.set_ylim(0.7,1.3)
        spavbl.grid('on')
        spavbl.set_xlabel('Time [h]')

    # Plot energy
    spenergy.plot((energy.t_stamps-t_ref)/3600., energy.energy/1e3, c='black', lw=2.)#, alpha=0.1)

    # Plot Intensity
    for beam_n in (1,2):
        if flag_fbct:
            sptotint.plot((fbct_bx[beam_n].t_stamps-t_ref)/3600., fbct_bx[beam_n].totint, '.--', color=colstr[beam_n])
        sptotint.plot((bct_bx[beam_n].t_stamps-t_ref)/3600., bct_bx[beam_n].values, '-', color=colstr[beam_n], lw=2.)

        if flag_bunch_length:
            spavbl.plot((blength_bx[beam_n].t_stamps-t_ref)/3600., blength_bx[beam_n].avblen/1e-9, '.-', color=colstr[beam_n])

    # Bunch length and intensity
    t_bl_1 = blength_bx[1].t_stamps
    mask_bl_he_1 = t_bl_1>dict_fill_bmodes[filln]['t_stop_SQUEEZE']

    total_bint = (bct_bx[1].interp(t_bl_1) + bct_bx[2].interp(t_bl_1))/ (2*n_bunches)
    total_bint = tm.make_timber_variable_list(t_bl_1[mask_bl_he_1], total_bint[mask_bl_he_1])
    bl_1 = blength_bx[1].avblen
    bl_2 = np.interp(t_bl_1, blength_bx[2].t_stamps, blength_bx[2].avblen)
    av_blen = (bl_1 + bl_2)/(2*1e-9)
    av_blen = tm.make_timber_variable_list(t_bl_1[mask_bl_he_1], av_blen[mask_bl_he_1])

    sp_blen_vs_int.plot(total_bint.values, av_blen.values, '.', lw=2., label=filln)

    if args.make_table_blen != None:
        table_blen = args.make_table_blen
        tt_blen = av_blen.nearest_t_stamp(table_blen)
    elif args.make_table_time != None:
        tt_blen = t_ref + args.make_table_time * 3600
        table_blen = av_blen.nearest_older_sample(tt_blen)
    if args.make_table_bint != None:
        table_bint = args.make_table_bint *1e11
        tt_bint = total_bint.nearest_t_stamp(table_bint)
        table_bint = total_bint.nearest_older_sample(tt_bint)
        table_blen = av_blen.nearest_older_sample(tt_bint)
        tt_blen = av_blen.nearest_t_stamp(table_blen)
    elif args.make_table_blen != None or args.make_table_time != None:
        # tt_blen, table_blen defined above
        table_bint = total_bint.nearest_older_sample(tt_blen)

    ii = 0
    group_names = dict_hl_groups.keys()
    group_name = group_names[ii]
    pl.suptitle(' Fill. %d started on %s\n%s'%(filln, tref_string, group_name))
    fig_h.canvas.set_window_title(group_name + ' %i' % filln)

    hl_var_names = dict_hl_groups[group_name][:]
    hl_var_names_copy = dict_hl_groups[group_name][:]
    for varname in hl_var_names_copy:
        if varname in blacklist:
            hl_var_names.remove(varname)

    heatloads = SetOfHomogeneousNumericVariables(variable_list=hl_var_names, timber_variables=fill_dict)

    # Model heat load
    if use_recalculated:
        hl_imped_fill = fc.HeatLoad_calculated_fill(fill_dict, hli_calculator, bct_dict=bct_bx, fbct_dict=fbct_bx, blength_dict=blength_bx)
        hl_sr_fill = fc.HeatLoad_calculated_fill(fill_dict, hlsr_calculator, bct_dict=bct_bx, fbct_dict=fbct_bx, blength_dict=blength_bx)
        hl_total_model = (hl_imped_fill.heat_load_calculated_total + hl_sr_fill.heat_load_calculated_total)*53.45
        hl_model_t_stamps = (hl_imped_fill.t_stamps - hl_imped_fill.t_stamps[0])/3600.
    else:
        hl_model = SetOfHomogeneousNumericVariables(variable_list=HL.variable_lists_heatloads['MODEL'], timber_variables=fill_dict)

    # CORRECT ARC AVERAGES
    if not use_recalculated and group_name == 'Arcs' and filln < first_correct_filln:
        hl_corr_factors = []
        for ii, varname in enumerate(dict_hl_groups[group_name]):
            if varname not in blacklist:
                hl_corr_factors.append(arc_correction_factor_list[ii])
        heatloads.correct_values(hl_corr_factors)

    if flag_average:
        hl_ts_curr, hl_aver_curr  = heatloads.mean()

    # Loop for arcs
    fill_arc_hl_dict[filln] = {}
    for ii, kk in enumerate(heatloads.variable_list):
        colorcurr = ms.colorprog(ii, heatloads.variable_list)

        # offset
        if t_zero is not None:
            offset = np.interp(t_ref+t_zero*3600, heatloads.timber_variables[kk].t_stamps, heatloads.timber_variables[kk].values)
        else:
            offset = 0.

        label = ''
        for st in kk.split('.POSST')[0].split('_'):
            if 'QRL' in st or 'QBS' in st or 'AVG' in st or 'ARC' in st:
                pass
            else:
                label += st + ' '
        label = label[:-1]

        sphlcell.plot((heatloads.timber_variables[kk].t_stamps-t_ref)/3600., heatloads.timber_variables[kk].values-offset,
            '-', color=colorcurr, lw=2., label=label)

        t_hl = heatloads.timber_variables[kk].t_stamps
        mask_he = t_hl > dict_fill_bmodes[filln]['t_stop_SQUEEZE']

        if use_recalculated:
            subtract = np.interp(t_hl[mask_he], hl_imped_fill.t_stamps, hl_total_model)
        else:
            subtract = np.interp(t_hl[mask_he], hl_model.timber_variables['LHC.QBS_CALCULATED_ARC.TOTAL'].t_stamps, hl_model.timber_variables['LHC.QBS_CALCULATED_ARC.TOTAL'].values)


        try: marker = ['.', 'x', 'o', 'v', '^'][i_fill]
        except: marker = '.'
        marker = '.'

        if i_fill == 0:
            sp_vs_int_label = label
            imp_sr_label = 'Imp. + SR'
        else:
            sp_vs_int_label = None
            imp_sr_label = None
        xx = total_bint.interp(t_hl[mask_he])
        yy = heatloads.timber_variables[kk].values[mask_he]-offset-subtract
        mask = xx > 0.3e11
        xx = xx[mask]
        yy = yy[mask]
        spvsint.plot(xx, yy, marker, color=colorcurr, lw=2., label=sp_vs_int_label)
        if ii == len(heatloads.variable_list)-1:
            spvsint.plot(xx, subtract[mask], '.', color='black', label=imp_sr_label)

        if make_table != None:
            index_xx = np.argmin(np.abs(xx - table_bint))
            table_hl = yy[index_xx]
            fill_arc_hl_dict[filln][kk] = (table_bint, table_hl, table_blen)
            if ii == len(heatloads.variable_list)-1:
                #Subplots for this fill
                sptotint.axvline((tt_blen - t_ref)/3600., color='black', label='Reference time', lw=2)
                sphlcell.axvline((tt_blen - t_ref)/3600., color='black', label='Reference time', lw=2)
                if flag_bunch_length:
                    spavbl.axvline((tt_blen - t_ref)/3600., color='black', label='Reference time', lw=2)
                # Shared subplots
                sp_blen_vs_int.axhline(table_blen, lw=2., color='black', label=None)
                spvsint.axvline(table_bint, lw=2., color='black', label=None)#'Reference int %i' % filln)

    # Model heat load
    if plot_model and group_name == 'Arcs':
        if use_recalculated:
            xx, yy = hl_model_t_stamps, hl_total_model
        else:
            kk = 'LHC.QBS_CALCULATED_ARC.TOTAL'
            xx, yy = (hl_model.timber_variables[kk].t_stamps-t_ref)/3600, hl_model.timber_variables[kk].values
        sphlcell.plot(xx,yy, '--', color='grey', lw=2., label='Imp.+SR')

    if flag_average:
        if t_zero is not None:
            offset = np.interp(t_ref+t_zero*3600, hl_ts_curr, hl_aver_curr)
        else:
            offset = 0.
        sphlcell.plot((hl_ts_curr-t_ref)/3600., hl_aver_curr-offset, 'k', lw=2)

    sphlcell.legend(prop={'size':myfontsz}, bbox_to_anchor=(1.05, 1),  loc='upper left')

spvsint.set_xlim(0.5e11, 1.1e11)
spvsint.set_ylim(0, None)
spvsint.legend(prop={'size':myfontsz}, bbox_to_anchor=(1, 1),  loc='upper left')

if args.filln_list == [5219, 5222, 5223]:
    sp_blen_vs_int.set_xlim(0.6e11, 1.1e11)
    sp_blen_vs_int.set_ylim(1.0, 1.3)
else:
    sp_blen_vs_int.set_xlim(0.6e11, 1.3e11)
    sp_blen_vs_int.set_ylim(0.7, 1.3)
sp_blen_vs_int.legend(prop={'size':myfontsz}, bbox_to_anchor=(1., 1),  loc='upper left')

pl.subplots_adjust(right=0.8, wspace=0.50)

#fig_vs_int.savefig('hl_vs_int_fill%s'%(fills_string), dpi=200)

#~ fig_blen_vs_int.set_size_inches(15., 8.)
#fig_blen_vs_int.subplots_adjust(right=0.7, wspace=0.30, bottom=.12, top=.87)
#fig_blen_vs_int.savefig('blen_vs_int_fill%s'%(fills_string), dpi=200)

if savefig:
    sf.saveall_pdijksta()

if make_table != None:
    leader = '\t' * 2
    fill_0 = filln_list[0]
    first_row = leader + r'Sector & %i ' % fill_0
    for filln in filln_list[1:]:
        first_row += '& %i & Ratio ' % filln
    first_row += r'\\\hline\hline'
    print(first_row)

    for kk in sorted(fill_arc_hl_dict[fill_0].keys()):
        hl_0 = fill_arc_hl_dict[fill_0][kk][1]
        row = leader + r'%s & %.0f ' % (kk[:3], hl_0)
        for filln in filln_list[1:]:
            hl = fill_arc_hl_dict[filln][kk][1]
            row += r'& %.0f & %.2f ' % (hl, (hl/hl_0))
        row += r'\\\hline'
        print(row)

pl.show()
