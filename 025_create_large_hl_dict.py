import sys
import cPickle
import re
import time
import os

import numpy as np

import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.LHC_Heatloads as hl
from LHCMeasurementTools.LHC_FBCT import FBCT
from LHCMeasurementTools.LHC_BCT import BCT
from LHCMeasurementTools.LHC_BQM import blength
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables
from LHCMeasurementTools.LHC_Energy import energy

import HeatLoadCalculators.impedance_heatload as hli
import HeatLoadCalculators.synchrotron_radiation_heatload as hls

# Config
pkl_file_name = './large_heat_load_dict2.pkl'
fills_bmodes_file = './fills_and_bmodes.pkl'
filling_pattern_csv = './fill_basic_data_csvs/injection_scheme.csv'
subtract_offset = True
hrs_after_squeeze = 24

if __name__ == '__main__' and os.path.isfile(pkl_file_name):
    raise ValueError('Pkl file already exists!')

##
re_bpi = re.compile('_(\d+)bpi')
filling_pattern_raw = tm.parse_timber_file(filling_pattern_csv, verbose=False)
key = filling_pattern_raw.keys()[0]
filling_pattern_ob = filling_pattern_raw[key]

# Proper keys for the output dictionary
re_arc = re.compile('(S\d\d)_QBS_AVG_ARC.POSST')
re_quad = re.compile('(Q)RL\w\w_0(\d[LR]\d)_QBS\d{3}.POSST')
re_special = re.compile('(Q)RLAA_(\d{2}\w\d)_QBS\d{3}(_\w\w)?.POSST')
re_list = [re_arc, re_quad, re_special]

def output_key(input_key, verbose=False, strict=True):
    for regex in re_list:
        info = re.search(regex, input_key)
        if info is not None:
            if verbose:
                print(var, ''.join(info.groups()))
            return ''.join(info.groups())
    else:
        if strict:
            raise ValueError('No match for %s' % var)
        else:
            return input_key

output_dict = {}
def add_to_output_dict(value, keys, zero=False):
    if zero:
        value = 0
    this_dict = output_dict
    for nn, key in enumerate(keys):
        if nn == len(keys)-1:
            if key not in this_dict:
                this_dict[key] = [value]
            else:
                this_dict[key].append(value)
        else:
            if key not in this_dict:
                this_dict[key] = {}
            this_dict = this_dict[key]

# Time keys
time_key_list = ['start_ramp', 'stop_squeeze']
for ii in xrange(hrs_after_squeeze):
    time_key_list.append('%ihrs' % (ii+1))

# Heat load groups
groups_dict = hl.groups_dict()
all_heat_load_vars = []
for group, varlist in groups_dict.iteritems():
    all_heat_load_vars += varlist
hl_var_dict = {}
for var in all_heat_load_vars:
    hl_var_dict[var] = {'key': output_key(var, strict=True)}

# Filling numbers
with open(fills_bmodes_file, 'r') as f:
    fills_and_bmodes = cPickle.load(f)
fills_0 = fills_and_bmodes.keys()
fills_0.sort()

# Model heat load calculators
imp_calc = hli.HeatLoadCalculatorImpedanceLHCArc()
sr_calc = hls.HeatLoadCalculatorSynchrotronRadiationLHCArc()

if __name__ == '__main__':
    # Main loop
    for ff, filln in enumerate(fills_0):
        t_stop_squeeze = fills_and_bmodes[filln]['t_stop_SQUEEZE']
        if t_stop_squeeze == -1:
            print('Fill %i did not reach the end of squeeze' % filln)
        else:
            print('Fill %i is being processed' % filln)
            try:
                fill_dict = {}
                fill_dict.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv'%filln, verbose=False))
                fill_dict.update(tm.parse_timber_file('fill_heatload_data_csvs/heatloads_fill_%d.csv'%filln, verbose=False))
                fill_dict.update(tm.parse_timber_file('fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_%d.csv'%filln, verbose=False))
            except IOError as e:
                print('Fill %i is skipped: %s!' % (filln,e))
                continue

            # Allocate objects that are used later
            en_ob = energy(fill_dict, beam=1)
            heatloads = SetOfHomogeneousNumericVariables(all_heat_load_vars, fill_dict)
            if subtract_offset:
                offset_dict = {}
                t_begin_inj = fills_and_bmodes[filln]['t_start_INJPROT']
                if t_begin_inj == -1:
                    print('Warning: Offset for fill %i could not be calculated as t_start_INJPROT is not in the fills and bmodes file!' % filln)
                    this_subtract_offset = False
                else:
                    this_subtract_offset = True
                    for var in all_heat_load_vars:
                        hl_ob = heatloads.timber_variables[var]
                        offset_dict[var] = hl_ob.calc_avg(t_begin_inj, t_begin_inj+600)
            bct_bx = {}
            blength_bx = {}
            fbct_bx = {}
            for beam_n in (1,2):
                bct_bx[beam_n] = BCT(fill_dict, beam=beam_n)
                blength_bx[beam_n] = blength(fill_dict, beam=beam_n)
                fbct_bx[beam_n] = FBCT(fill_dict, beam=beam_n)

            # Fill Number
            add_to_output_dict(filln, ['filln'])

            # Filling pattern and bpi
            pattern = filling_pattern_ob.nearest_older_sample(t_stop_squeeze)[0]
            add_to_output_dict(pattern, ['filling_pattern'])
            bpi_info = re.search(re_bpi, pattern)
            if bpi_info is not None:
                bpi = int(bpi_info.group(1))
            else:
                bpi = -1
            add_to_output_dict(bpi, ['bpi'])

            # Energy, only one per fill
            fill_energy = en_ob.nearest_older_sample(t_stop_squeeze)*1e9
            add_to_output_dict(fill_energy, ['energy'])

            # subloop for time points
            t_start_ramp = fills_and_bmodes[filln]['t_start_RAMP']
            end_time = fills_and_bmodes[filln]['t_endfill']
            for kk, time_key in enumerate(time_key_list):
                if kk == 0:
                    tt = t_start_ramp
                elif kk == 1:
                    tt = t_stop_squeeze
                else:
                    tt = t_stop_squeeze + (kk-1)*3600
                if tt > end_time:
                    zero=True
                else:
                    zero=False

                this_add_to_dict = lambda x, keys: add_to_output_dict(x, [time_key]+keys, zero=zero)

                # t_stamps
                this_add_to_dict(tt, ['t_stamps'])

                # intensity
                tot_int = 0
                int_bx = {}
                for beam in (1,2):
                    if zero:
                        this_int = 0
                    else:
                        this_int = float(bct_bx[beam].nearest_older_sample(tt))
                    this_add_to_dict(this_int, ['intensity', 'b%i' % beam])
                    int_bx[beam] = this_int
                    tot_int += this_int
                this_add_to_dict(tot_int, ['intensity', 'total'])

                # Bunch length
                tot_avg, tot_var = 0, 0
                this_blength_bx = {}
                for beam in (1,2):
                    if zero:
                        avg, sig = 0, 0
                    else:
                        all_blen = blength_bx[beam].nearest_older_sample(tt)
                        avg = np.mean(all_blen)
                        sig = np.std(all_blen)
                    this_blength_bx[beam] = avg
                    this_add_to_dict(avg, ['blength', 'b%i' % beam, 'avg'])
                    this_add_to_dict(sig, ['blength', 'b%i' % beam, 'sig'])
                    tot_avg += avg
                    tot_var += sig**2
                this_add_to_dict(tot_avg, ['blength', 'total', 'avg'])
                this_add_to_dict(np.sqrt(0.5*tot_var), ['blength', 'total', 'sig'])

                # Number of bunches
                n_bunches_bx = {}
                for beam in (1,2):
                    if zero:
                        n_bunches = 0
                    else:
                        bint = fbct_bx[beam].nearest_older_sample(tt)
                        min_int = 0.1 * max(bint)
                        mask_filled = bint > min_int
                    n_bunches = sum(mask_filled)
                    n_bunches_bx[beam] = n_bunches
                    this_add_to_dict(n_bunches, ['n_bunches', 'b%i' % beam])

                # Imp / SR
                tot_imp, tot_sr, tot_model = 0, 0, 0
                for beam in (1,2):
                    beam_int = int_bx[beam]
                    n_bunches = n_bunches_bx[beam]
                    this_blength = this_blength_bx[beam]
                    if n_bunches != 0 and this_blength != 0 and not zero:
                        imp = imp_calc.calculate_P_Wm(beam_int/n_bunches, this_blength, fill_energy, n_bunches)
                        sr = sr_calc.calculate_P_Wm(beam_int/n_bunches, this_blength, fill_energy, n_bunches)
                    else:
                        imp, sr = 0, 0
                    tot_imp += imp
                    tot_model += imp
                    this_add_to_dict(imp, ['heat_load', 'imp', 'b%i' % beam])
                    tot_sr += sr
                    tot_model += sr
                    this_add_to_dict(sr, ['heat_load', 'sr', 'b%i' % beam])
                this_add_to_dict(tot_imp, ['heat_load', 'imp', 'total'])
                this_add_to_dict(tot_sr, ['heat_load', 'sr', 'total'])
                this_add_to_dict(tot_model, ['heat_load', 'total_model'])

                # Heat loads
                for var in all_heat_load_vars:
                    hl_ob = heatloads.timber_variables[var]
                    if this_subtract_offset:
                        offset = offset_dict[var]
                    else:
                        offset = 0
                    if not zero:
                        hl = hl_ob.nearest_older_sample(tt) - offset
                    else:
                        hl = 0
                    key = hl_var_dict[var]['key']
                    this_add_to_dict(hl, ['heat_load', key])

    # Dump this dict
    with open(pkl_file_name, 'w') as f:
        cPickle.dump(output_dict, f, protocol=-1)
