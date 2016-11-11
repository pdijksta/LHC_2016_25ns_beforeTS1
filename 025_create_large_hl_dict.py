import sys
import cPickle
import re
import time

import numpy as np

import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.LHC_Heatloads as hl
from LHCMeasurementTools.LHC_FBCT import FBCT
from LHCMeasurementTools.LHC_BCT import BCT
from LHCMeasurementTools.LHC_BQM import filled_buckets, blength
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables
from LHCMeasurementTools.LHC_Energy import energy

import HeatLoadCalculators.impedance_heatload as hli
import HeatLoadCalculators.synchrotron_radiation_heatload as hls

# Config
pkl_file_name = './large_heat_load_dict.pkl'
fills_bmodes_file = './fills_and_bmodes.pkl'
filling_pattern_csv = './fill_basic_data_csvs/injection_scheme.csv'
subtract_offset = True

##
re_bpi = re.compile('_(\d+)bpi')
filling_pattern_raw = tm.parse_timber_file(filling_pattern_csv, verbose=False)
key = filling_pattern_raw.keys()[0]
filling_pattern_ob = filling_pattern_raw[key]

# TODO: proper keys for the output dictionary
re_arc = re.compile('(S\d\d)_QBS_AVG_ARC.POSST')

# Heat load groups
groups_dict = hl.groups_dict()
all_heat_load_vars = []
for group, varlist in groups_dict.iteritems():
    all_heat_load_vars += varlist

# Filling numbers
with open(fills_bmodes_file, 'r') as f:
    fills_and_bmodes = cPickle.load(f)
fills_0 = fills_and_bmodes.keys()
fills_0.sort()

# Model heat load calculators
imp_calc = hli.HeatLoadCalculatorImpedanceLHCArc()
sr_calc = hls.HeatLoadCalculatorSynchrotronRadiationLHCArc()

# Initialize output arrays. First item is the description
fills = ['filln']
filling_pattern = ['filling_pattern']
bpi = ['bpi']
n_bunches_list = ['N_bunches']
hl_time_points = ['time_points']
tot_int = ['tot_int']
b1_int = ['B1_int']
b2_int = ['B2_int']
energy_list = ['energy']

blen_dict = {}
for prefix in ('b1_','b2_',''):
    for suffix in ['avg', 'sig']:
        blen_dict[prefix+'blen_'+suffix] = []

# Output arrays for all the heat load variables
class Dummy:
    pass
all_hl_lists_ob = Dummy()
all_hl_lists_ob.impedance = ['Impedance']
all_hl_lists_ob.synchRad = ['SynchRad']
for var in all_heat_load_vars:
    setattr(all_hl_lists_ob, var, [var])

var_list = [fills, filling_pattern, bpi, n_bunches_list]
nested_var_list = [hl_time_points, b1_int, b2_int, tot_int, energy_list]

# Main loop
for ff, filln in enumerate(fills_0[:50]):
    t_stop_squeeze = fills_and_bmodes[filln]['t_stop_SQUEEZE']
    if t_stop_squeeze == -1:
        print('This fill did not reach the end of squeeze: %i' % filln)
    else:

        # Fill Number
        fills.append(filln)

        # Filling pattern and bpi
        pattern = filling_pattern_ob.nearest_older_sample(t_stop_squeeze)[0]
        filling_pattern.append(pattern)
        bpi_info = re.search(re_bpi, pattern)
        if bpi_info is not None:
            bpi.append(int(bpi_info.group(1)))
        else:
            bpi.append(-1)

        fill_dict = {}
        fill_dict.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv'%filln, verbose=False))
        fill_dict.update(tm.parse_timber_file('fill_heatload_data_csvs/heatloads_fill_%d.csv'%filln, verbose=False))
        fill_dict.update(tm.parse_timber_file('fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_%d.csv'%filln, verbose=False))

        # Actual numbers of bunches from BQM
        filled_buckets_1 = filled_buckets(fill_dict, beam=1)
        filled_buckets_2 = filled_buckets(fill_dict, beam=2)
        n_bunches_1 = float(max(filled_buckets_1.Nbun))
        n_bunches_2 = float(max(filled_buckets_2.Nbun))
        n_bunches = max(n_bunches_1, n_bunches_2)
        n_bunches_list.append(n_bunches)
        if n_bunches_1 != n_bunches_2:
            print('Fill %i: N bunches for beam 1: %i, for beam 2: %i, choosing %i' % (filln, n_bunches_1, n_bunches_2, n_bunches))

        # get time points
        time_points = []
        t_start_ramp = fills_and_bmodes[filln]['t_start_RAMP']
        time_points.append(t_start_ramp)

        end_time = fills_and_bmodes[filln]['t_endfill']
        t = t_stop_squeeze
        while t < end_time:
            time_points.append(t)
            t += 3600
        hl_time_points.append(time_points)

        # intensity
        this_tot_int = []
        this_b1_int = []
        this_b2_int = []
        bct_bx = {}
        for beam_n in (1,2):
            bct_bx[beam_n] = BCT(fill_dict, beam=beam_n)
        for tt in time_points:
            int_b1 = float(bct_bx[1].nearest_older_sample(tt))
            int_b2 = float(bct_bx[2].nearest_older_sample(tt))
            this_tot_int.append(int_b1+int_b2)
            this_b1_int.append(int_b1)
            this_b2_int.append(int_b2)
        b1_int.append(this_b1_int)
        b2_int.append(this_b2_int)
        tot_int.append(this_tot_int)

        # energy
        en_ob = energy(fill_dict, beam=1)
        fill_energy = en_ob.nearest_older_sample(t_stop_squeeze)*1e9
        energy_list.append(fill_energy)

        # bunch length
        blength_bx = {}
        for beam_n in (1,2):
            blength_bx[beam_n] = blength(fill_dict, beam=beam_n)

        this_b1_blen_avg = []
        this_b1_blen_sig = []
        this_b2_blen_avg = []
        this_b2_blen_sig = []
        this_blen_avg = []
        this_blen_sig = []
        for tt in time_points:
            all_blen_1 = blength_bx[1].nearest_older_sample(tt)
            all_blen_2 = blength_bx[2].nearest_older_sample(tt)
            avg_1 = np.mean(all_blen_1)
            sig_1 = np.std(all_blen_1)
            avg_2 = np.mean(all_blen_2)
            sig_2 = np.std(all_blen_2)
            this_b1_blen_avg.append(avg_1)
            this_b2_blen_avg.append(avg_2)
            this_b1_blen_sig.append(sig_1)
            this_b2_blen_sig.append(sig_2)
            this_blen_avg.append(np.mean([avg_1, avg_2]))
            this_blen_sig.append(np.sqrt(0.5*(sig_1**2 + sig_2**2)))
        glob = globals()
        for key in blen_dict:
            blen_dict[key].append(glob['this_'+key])

        # Imp / SR
        this_imp = []
        this_sr = []
        for ii,tt in enumerate(time_points):
            int_b1 = this_b1_int[ii]
            blen_b1 = this_b1_blen_avg[ii]/4.
            int_b2 = this_b2_int[ii]
            blen_b2 = this_b2_blen_avg[ii]/4.
            # Blength measurements might not always work
            if blen_b1 == 0. or blen_b2 == 0.:
                break

            imp_b1 = imp_calc.calculate_P_Wm(int_b1/n_bunches, blen_b1, fill_energy, n_bunches)
            imp_b2 = imp_calc.calculate_P_Wm(int_b2/n_bunches, blen_b2, fill_energy, n_bunches)
            this_imp.append(imp_b1+imp_b2)
            sr1 = sr_calc.calculate_P_Wm(int_b1/n_bunches, blen_b1, fill_energy, n_bunches)
            sr1 = sr_calc.calculate_P_Wm(int_b2/n_bunches, blen_b2, fill_energy, n_bunches)
            this_sr.append(sr1+sr2)
        all_hl_lists_ob.impedance.append(this_imp)
        all_hl_lists_ob.synchRad.append(this_sr)

        # Heat loads
        heatloads = SetOfHomogeneousNumericVariables(all_heat_load_vars, fill_dict)
        for var in all_heat_load_vars:
            hl_ob = heatloads.timber_variables[var]
            hl_list = []
            if subtract_offset:
                t_begin_inj = fills_and_bmodes[filln]['t_start_INJPROT']
                if t_begin_inj == -1:
                    print('Warning: Offset could not be calculated as t_start_INJPROT is not in the fills and bmodes file!')
                    offset = 0.
                else:
                    offset = hl_ob.calc_avg(t_begin_inj, t_begin_inj+600)
            else:
                offset = 0.
            for tt in time_points:
                hl = hl_ob.nearest_older_sample(tt) - offset
                hl_list.append(hl)
            getattr(all_hl_lists_ob, var).append(hl_list)


# populate output dict
output_dict = {}
for key in all_hl_lists_ob.__dict__:
    member = getattr(all_hl_lists_ob, key)
    description = member.pop(0)
    output_dict[description] = member
for ls in var_list + nested_var_list:
    description = ls.pop(0)
    output_dict[description] = ls

# TODO
nested_ob = Dummy()
for var in nested_var_list:
    var_1 = var+

output_dict['b_length'] = blen_dict

## Dump this dict
#with open(pkl_file_name, 'w') as f:
#    cPickle.dump(output_dict, f, protocol=-1)

