# This Script is intended to be used for PyEcloud benchmark simulations.
# The heat load on the arc is calculated using LHC data for a given fill and time.
# Comparing these to the heatloads from PyEcloud simulations will then allow for 
#   an estimation of the SEY parameter.
# It has been extended to also list the heat load on the Q6 quadrupoles.

# Written by Philipp Dijkstal, philipp.dijkstal@cern.ch

import sys
import os
import cPickle  # it is recommended to use cPickle over pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse as arg
import re

import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.LHC_Energy as Energy
import LHCMeasurementTools.mystyle as ms
from LHCMeasurementTools.LHC_BCT import BCT
import LHCMeasurementTools.LHC_Heatloads as HL
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables
from LHCMeasurementTools.mystyle import colorprog

import HeatLoadCalculators.impedance_heatload as ihl
import HeatLoadCalculators.synchrotron_radiation_heatload as srhl 
import HeatLoadCalculators.FillCalculator as fc

# CONFIG
default_avg_period = 0.1  # in hours
default_offset_period_begin = 0.1
default_offset_period_end = 0.35

first_correct_filln = 4474
colstr = {}
colstr[1] = 'b'
colstr[2] = 'r'
beams_list = [1, 2]
myfontsz = 16
ms.mystyle_arial(fontsz=myfontsz, dist_tick_lab=8)
pickle_name = 'heatload_arcs.pkl'

# PARSE ARGS
parser = arg.ArgumentParser(description='Calculate the heat loads on all arcs at a specified time for a given fill.' +
     'An average is taken. The heat loads should be stable at this point in time.')

parser.add_argument('fill', metavar='FILL', type=int, help='LHC fill number, must be at least %i.' % first_correct_filln)
parser.add_argument('time', metavar='TIME', type=str, help='Time after the specified FILL number has been set.\n Obtain it with the 016 script.', nargs='+')
parser.add_argument('-a', metavar='AVG_PERIOD', type=float, default=default_avg_period, help='The time in hours this program uses to find an average around the specified TIME.\nDefault: +/- %.2f' % default_avg_period)
parser.add_argument('-p', help='Save heat loads to pickle if entry does not exist.', action='store_true')
parser.add_argument('-f', help='Overwrite heat loads at the pickle if the entry does already exist.', action='store_true')
parser.add_argument('-n', help='No plot will be shown.', action='store_false')
parser.add_argument('-o', metavar=('T1', 'T2'), nargs=2, type=float, default=[default_offset_period_begin, default_offset_period_end], help='The time in hours this program uses to calculate an offset.\nDefault: %.2f - %.2f' % (default_offset_period_begin, default_offset_period_end))
parser.add_argument('-c', help='Show contribution from different devices', action='store_true')

args = parser.parse_args()
avg_period = args.a
filln = args.fill
time_of_interest_str_arr = args.time
store_pickle = args.p or args.f
overwrite_pickle = args.f
show_plot = args.n or args.c
show_heatload_contributions = args.c
[offset_time_hrs_begin, offset_time_hrs_end] = args.o

time_of_interest_arr = [float(time_of_interest_str) for time_of_interest_str in time_of_interest_str_arr]

main_key_dict = {}
for time_of_interest in time_of_interest_arr:
    main_key_dict[time_of_interest] = str(filln) + ' ' + str(time_of_interest)

# A minimum fill number is inferred from the 016 script
if filln < first_correct_filln:
    raise ValueError("Fill number too small. Look at the help for this script.")

# The heat loads calculated during this call of the script are stored here
this_hl_dict = {}
this_hl_dict_options = {\
        'Offset' : [offset_time_hrs_begin, offset_time_hrs_end],
        'Avg_period' : avg_period,
        'Time' : time_of_interest_str_arr
        }

# LOAD DATA
dict_hl_groups = {}
arc_keys_list = HL.variable_lists_heatloads['AVG_ARC']
quad_keys_list = HL.variable_lists_heatloads['Q6s_IR1'] \
        + HL.variable_lists_heatloads['Q6s_IR5'] \
        + HL.variable_lists_heatloads['Q6s_IR2'] \
        + HL.variable_lists_heatloads['Q6s_IR8'] 
imp_label = 'Imp'
sr_label = 'SR'
model_label = imp_label + '+' + sr_label

all_keys = arc_keys_list + quad_keys_list

with open('fills_and_bmodes.pkl', 'rb') as fid:
    dict_fill_bmodes = cPickle.load(fid)

# Saves time when repeatedly running from ipython
fill_dict = {}
fill_dict.update(tm.parse_timber_file('./fill_basic_data_csvs/basic_data_fill_%d.csv' % filln, verbose=False))
fill_dict.update(tm.parse_timber_file('./fill_heatload_data_csvs/heatloads_fill_%d.csv' % filln, verbose=False))
fill_dict.update(tm.parse_timber_file('./fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_%d.csv' % filln, verbose=False))

bct_bx = {}
for beam_n in beams_list:
    bct_bx[beam_n] = BCT(fill_dict, beam=beam_n)

energy = Energy.energy(fill_dict, beam=1)
t_ref = dict_fill_bmodes[filln]['t_startfill']
tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_ref))

heatloads = SetOfHomogeneousNumericVariables(variable_list=all_keys, timber_variables=fill_dict)

def get_output_key(input_key):
    """ Map keys for output dictionary. """
    if input_key in arc_keys_list:
        return input_key[0:3]
    elif input_key in quad_keys_list:
        return 'Q' + input_key[6:10]
    else:
        return input_key

def cut_arrays(arr, time_of_interest,avg_period=avg_period):
    """
    Expects two column input: time, values.
    Returns the values for which time-avg_period < time < time+avg_period is satisfied
    """
    begin = time_of_interest-avg_period
    end = time_of_interest+avg_period

    condition = np.logical_and(begin < arr[:,0], arr[:,0] < end)
    return np.extract(condition, arr[:,1])

def get_heat_loads(key):
    """
    Gets the heatloads of the specified key through the timber variables module.
    """
    time_heatload = np.array([(heatloads.timber_variables[key].t_stamps-t_ref)/3600., heatloads.timber_variables[key].values]).T
    avg_heatload = []
    avg_heatload_sigma = []
    avg_offset = []
    for time_of_interest in time_of_interest_arr:
        time_heatload_cut = cut_arrays(time_heatload, time_of_interest)
        avg_heatload.append(np.mean(time_heatload_cut))
        avg_heatload_sigma.append(np.std(time_heatload_cut))
        
        offset_avg_period = (offset_time_hrs_end - offset_time_hrs_begin)/2
        offset_time = offset_time_hrs_begin + offset_avg_period
        
        offset_cut = cut_arrays(time_heatload, offset_time, avg_period=offset_avg_period)
        avg_offset.append(np.mean(offset_cut))

    return [avg_heatload, avg_heatload_sigma, avg_offset]

def add_to_output_dict(main_key, input_key, avg_heatload, avg_heatload_sigma, offset):
    output_key = get_output_key(input_key)
    #print("The heatload %s is\n%.2f\t%.2f\twith offset %.2f\n" % (output_key, avg_heatload, avg_heatload_sigma,offset))
    if main_key not in this_hl_dict:
        this_hl_dict[main_key] = {}
    if 'Options' not in this_hl_dict[main_key]:
        this_hl_dict[main_key]['Options'] = this_hl_dict_options

    this_hl_dict[main_key][output_key] = {\
            'Heat_load': avg_heatload, 
            'Sigma': avg_heatload_sigma,
            'Offset': offset
            }


hli_calculator  = ihl.HeatLoadCalculatorImpedanceLHCArc()
hlsr_calculator  = srhl.HeatLoadCalculatorSynchrotronRadiationLHCArc()

hl_imped_fill = fc.HeatLoad_calculated_fill(fill_dict, hli_calculator)
hl_sr_fill = fc.HeatLoad_calculated_fill(fill_dict, hlsr_calculator)

imp_data = np.array([(hl_imped_fill.t_stamps-t_ref)/3600, hl_imped_fill.heat_load_calculated_total]).T
sr_data = np.array([(hl_sr_fill.t_stamps-t_ref)/3600, hl_sr_fill.heat_load_calculated_total]).T

for key in arc_keys_list + quad_keys_list:
    [hl_arr, sigma_arr, offset_arr] = get_heat_loads(key)
    for ctr, time_of_interest in enumerate(time_of_interest_arr):
        main_key = main_key_dict[time_of_interest]
        add_to_output_dict(main_key, key, hl_arr[ctr], sigma_arr[ctr], offset_arr[ctr])
    
for ctr, time_of_interest in enumerate(time_of_interest_arr):
    cut_imp = cut_arrays(imp_data, time_of_interest, avg_period=avg_period)
    cut_sr = cut_arrays(sr_data, time_of_interest, avg_period=avg_period)

    avg_imp = np.mean(cut_imp)*HL.magnet_length['special_total'][0]
    sigma_imp = np.std(cut_imp)*HL.magnet_length['special_total'][0]
    avg_sr = np.mean(cut_sr)*HL.magnet_length['special_total'][0]
    sigma_sr = np.std(cut_sr)*HL.magnet_length['special_total'][0]
    main_key = main_key_dict[time_of_interest]
    add_to_output_dict(main_key, imp_label, avg_imp, sigma_imp, 0)
    add_to_output_dict(main_key, sr_label, avg_sr, sigma_sr, 0)
    add_to_output_dict(main_key, 'Imp+SR', avg_imp+avg_sr, sigma_imp+sigma_sr, 0)


    # SAVE PICKLE

if store_pickle:
    print('Storing into pickle.')
    if not os.path.isfile(pickle_name):
        heatload_dict = {} 
    else :
        with open(pickle_name, 'r') as hl_dict_file:
            heatload_dict = cPickle.load(hl_dict_file)

    if overwrite_pickle:
        for key_new in this_hl_dict:
            if key_new in heatload_dict:
                del heatload_dict[key_new]
    
    filln_str = str(filln)
    t_o_i_str = str(time_of_interest)
    main_key = filln_str + ' ' + t_o_i_str

    heatload_dict.update(this_hl_dict)
    with open(pickle_name, 'w') as hl_dict_file:
        cPickle.dump(heatload_dict, hl_dict_file, protocol=-1)
    
#    if overwrite_pickle and 'test' in heatload_dict:
#        print('Deleting old entry.')
#        del heatload_dict['test']
#
#    if main_key not in heatload_dict:
#        heatload_dict[main_key] = this_hl_dict
#        with open(pickle_name, 'w') as hl_dict_file:
#            cPickle.dump(heatload_dict, hl_dict_file, protocol=-1)
#    else:
#        print('This entry already exists in the pickle, not storing!!\n')

# PLOTS
if show_plot:
    plt.close('all')
    fig = plt.figure()

    fig.patch.set_facecolor('w')
    fig.canvas.set_window_title('LHC Arcs and Q6')
    fig.set_size_inches(15., 8.)

    plt.suptitle(' Fill. %d started on %s\nLHC Arcs and Q6' % (filln, tref_string))
    plt.subplots_adjust(right=0.7, wspace=0.30)

    # Intensity and Energy
    sptotint = plt.subplot(3, 1, 1)
    sptotint.set_ylabel('Total intensity [p+]')
    sptotint.grid('on')
    for beam_n in beams_list:
        sptotint.plot((bct_bx[beam_n].t_stamps-t_ref)/3600., bct_bx[beam_n].values, '-', color=colstr[beam_n])

    spenergy = sptotint.twinx()
    spenergy.plot((energy.t_stamps-t_ref)/3600., energy.energy/1e3, c='black', lw=2.)  # alpha=0.1)
    spenergy.set_ylabel('Energy [TeV]')
    spenergy.set_ylim(0, 7)

    # Cell heat loads
    sphlcell = plt.subplot(3, 1, 2, sharex=sptotint)
    sphlcell.set_ylabel('Heat load [W]')
    sphlcell.grid('on')

    # Quad heat loads
    sphlquad = plt.subplot(3, 1, 3, sharex=sptotint)
    sphlquad.grid('on')
    sphlquad.set_ylabel('Heat load [W]')
    sphlquad.set_xlabel('Time [h]')

    # Heat loads arcs and quads
    arc_ctr, quad_ctr = 0, 0
    for key in arc_keys_list + quad_keys_list:
        output_key = get_output_key(key)
        if output_key[0] == 'Q':
            sp = sphlquad
            color = colorprog(quad_ctr, len(quad_keys_list))
            quad_ctr += 1
        else:
            sp = sphlcell
            color = colorprog(arc_ctr, len(arc_keys_list)+1)
            arc_ctr += 1
        yy_time = (heatloads.timber_variables[key].t_stamps-t_ref)/3600.
        ones = np.ones_like(yy_time)
        
        sp.plot(yy_time, heatloads.timber_variables[key].values, '-', lw=2., label=output_key, color=color)

    # Heat loads model
    sphlcell.plot(imp_data[:,0], imp_data[:,1]*HL.magnet_length['special_total'][0], label=imp_label)
    sphlcell.plot(sr_data[:,0], sr_data[:,1]*HL.magnet_length['special_total'][0], label=sr_label)
    sphlcell.plot(imp_data[:,0], (imp_data[:,1]+sr_data[:,1])*HL.magnet_length['special_total'][0], label='Imp+SR')


    sphlquad.legend(prop={'size': myfontsz}, bbox_to_anchor=(1.1, 1),  loc='upper left')
    sphlcell.legend(prop={'size': myfontsz}, bbox_to_anchor=(1.1, 1),  loc='upper left')

    # Vertical line to indicate time_of_interest
    for sp in [sphlcell, spenergy, sphlquad]:
        sp.axvline(offset_time_hrs_begin, color='black')
        sp.axvline(offset_time_hrs_end, color='black')
        for time_of_interest in time_of_interest_arr:
            if time_of_interest == 0:
                continue
            sp.axvline(time_of_interest, color='black')
            for xx in [time_of_interest - avg_period, time_of_interest + avg_period]:
                sp.axvline(xx, ls='--', color='black')


if show_heatload_contributions:

    re_quad_15 = re.compile('^Q06[LR][15]$')
    re_quad_28 = re.compile('^Q06[LR][28]$')
    len_q6_28 = HL.magnet_length['Q6s_IR2'][0]
    len_q6_15 = HL.magnet_length['Q6s_IR1'][0]
    len_cell = HL.magnet_length['special_total'][0]
    len_quad_cell = HL.magnet_length['special_HC_Q1'][0]

    # Create array for average quadrupole heat load
    summed_hl = 0
    summed_len = 0
    for key_ctr, key in enumerate(quad_keys_list):
        output_key = get_output_key(key)
        if re_quad_15.match(output_key):
            len_quad = len_q6_15
        elif re_quad_28.match(output_key):
            len_quad = len_q6_28
        else:
            raise ValueError('No match for key %s' % output_key)

        if key_ctr == 0:
            # Time stamps of first quad
            quad_time_ref = heatloads.timber_variables[key].t_stamps
            quad_time_ref_hrs = (quad_time_ref - t_ref)/3600.
            
            # Impedance per m and cell
            interp_imp = np.interp(quad_time_ref, hl_imped_fill.t_stamps, hl_imped_fill.heat_load_calculated_total)
            imp_cell = interp_imp * len_cell
            
            # SR per m and cell
            interp_sr = np.interp(quad_time_ref, hl_sr_fill.t_stamps, hl_sr_fill.heat_load_calculated_total)
            sr_cell = interp_sr * len_cell
        
        # Quad heat load, not normalized
        interp_hl = np.interp(quad_time_ref, heatloads.timber_variables[key].t_stamps, heatloads.timber_variables[key].values) - interp_imp * len_quad
        
        summed_hl += interp_hl
        summed_len += len_quad
    
    # Average quad heat load per cell
    avg_quad_hl = summed_hl/summed_len * len_quad_cell

    for key_ctr, key in enumerate(arc_keys_list):
        sp_ctr = key_ctr % 4 + 1

        if sp_ctr == 1:
            sp = None
            fig = plt.figure()
            title = 'Detailed arc heat loads'
            fig.canvas.set_window_title(title)
            plt.suptitle(title, fontsize=25)
            fig.patch.set_facecolor('w')

        sp = plt.subplot(2,2,sp_ctr, sharex=sp)
        sp.set_xlabel('Time [h]')
        sp.set_ylabel('Heat load [W/hc]')
        sp.set_title('Arc %s' % get_output_key(key))
        sp.grid('on')
        sp.set_ylim((0,sphlcell.get_ylim()[1]))

        sp2 = sp.twinx()
        sp2.set_ylabel('Energy [TeV]')
        sp2.plot((energy.t_stamps-t_ref)/3600., energy.energy/1e3, c='black', lw=2., ls='--', label='Energy')

        # Total
        yy_time = (heatloads.timber_variables[key].t_stamps-t_ref)/3600.
        sp.plot(yy_time, heatloads.timber_variables[key].values, label='Total', color='black', lw=3)

        # Imp
        bottom = np.zeros_like(quad_time_ref_hrs)
        top = np.copy(imp_cell)
        sp.fill_between(quad_time_ref_hrs, bottom, top, label='Impedance', alpha=0.5, color='r')
        bottom = np.copy(top)
        
        # SR
        top += sr_cell
        sp.fill_between(quad_time_ref_hrs, bottom, top, label='SR', alpha=0.5, color='b')
        bottom = np.copy(top)

        # Quad
        top += avg_quad_hl
        sp.fill_between(quad_time_ref_hrs, bottom, top, label='Average quad', alpha=0.5, color='orange')
        bottom = np.copy(top)

        # Rest
        bottom_rescaled = np.interp(yy_time, quad_time_ref_hrs, bottom)
        # Make sure the rest is not smaller than imp+sr+quad
        yy_all = np.copy(heatloads.timber_variables[key].values)
        mask_yy_too_small = yy_all < bottom_rescaled
        yy_all[mask_yy_too_small] = bottom_rescaled[mask_yy_too_small]
        sp.fill_between(yy_time, bottom_rescaled, yy_all, label='Dipoles and drifts', alpha=0.5, color='green')

        if sp_ctr == 2:
            sp.legend(bbox_to_anchor=(1.1, 1))


if show_plot:
    plt.show()
