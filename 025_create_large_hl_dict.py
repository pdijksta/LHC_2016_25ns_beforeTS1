import sys
import cPickle
import re
import time

import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.LHC_Heatloads as hl
from LHCMeasurementTools.LHC_FBCT import FBCT
from LHCMeasurementTools.LHC_BCT import BCT
from LHCMeasurementTools.LHC_BQM import filled_buckets, blength
from LHCMeasurementTools.SetOfHomogeneousVariables import SetOfHomogeneousNumericVariables

import HeatLoadCalculators.impedance_heatload
import HeatLoadCalculators.synchrotron_radiation_heatload

# Config
pkl_file_name = './large_heat_load_dict.pkl'

fills_bmodes_file = './fills_and_bmodes.pkl'
filling_pattern_csv = './fill_basic_data_csvs/injection_scheme.csv'

re_bpi = re.compile('_(\d+)bpi')

filling_pattern_raw = tm.parse_timber_file(filling_pattern_csv, verbose=False)
key = filling_pattern_raw.keys()[0]
filling_pattern_ob = filling_pattern_raw[key]

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

# Initialize output arrays. First item is the description
fills = ['Fill number']
filling_pattern = ['Filling pattern']
bpi = ['Bunches per injection']
n_bunches_list = ['Number of bunches']
hl_time_points = ['Time points with heat load. First point is start of ramp. Second point is end of squeeze. Further points are after one hour']

# Output arrays for all the heat load variables
class Dummy:
    pass
all_hl_lists_ob = Dummy()
all_hl_lists_ob.impedance = []
all_hl_lists_ob.synchRad = []
for var in all_heat_load_vars:
    setattr(all_hl_lists_ob, var, [var])


# Main loop
runs = 0
for ii,filln in enumerate(fills_0):

    t_stop_squeeze = fills_and_bmodes[filln]['t_stop_SQUEEZE']

    # Onay consider fills that are not empty
    if t_stop_squeeze != -1:

        #### 
        #remove later
        runs +=1
        if runs > 1:
            sys.exit()
        ####

        # Fill Number
        fills.append(filln)

        # Filling pattern and bpi
        pattern = filling_pattern_ob.nearest_older_sample(t_stop_squeeze)[0]
        filling_pattern.append(pattern)
        bpi_info = re.search(re_bpi, pattern)
        if bpi_info is not None:
            bpi.append(bpi_info.group(1))
        else:
            # Information cannot be inferred from pattern
            bpi.append('-1')

        fill_dict = {}
        fill_dict.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv'%filln, verbose=False))
        fill_dict.update(tm.parse_timber_file('fill_heatload_data_csvs/heatloads_fill_%d.csv'%filln, verbose=False))
        fill_dict.update(tm.parse_timber_file('fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_%d.csv'%filln, verbose=False))

        # Actual numbers of bunches from BQM
        # time_0 = time.time()
        filled_buckets_1 = filled_buckets(fill_dict, beam=1)
        filled_buckets_2 = filled_buckets(fill_dict, beam=2)
        # time_1 = time.time()
        # print("This took %fs" % (time_1 - time_0)) # around 0.3s
        n_bunches_1 = max(filled_buckets_1.Nbun)
        n_bunches_2 = max(filled_buckets_2.Nbun)
        n_bunches = max(n_bunches_1, n_bunches_2)
        n_bunches_list.append(n_bunches)
        if n_bunches_1 != n_bunches_2:
            print('Fill %i: N bunches for beam 1: %i, for beam 2: %i, choosing %i' % (filln, n_bunches_1, n_bunches_2, n_bunches))

        # get time points
        time_points = []
        time_points.append(fills_and_bmodes[filln]['t_start_RAMP'])

        end_time = fills_and_bmodes[filln]['t_endfill']
        t = t_stop_squeeze
        while t < end_time:
            time_points.append(t)
            t += 3600
        hl_time_points.append(time_points)

        # get heat loads
        heatloads = SetOfHomogeneousNumericVariables(all_heat_load_vars, fill_dict)
        for var in all_heat_load_vars:
            hl_ob = heatloads.timber_variables[var]
            hl_list = []
            # possibly calculate offsets here
            for tt in time_points:
                hl = hl_ob.nearest_older_sample(tt)
                hl_list.append(hl)
            getattr(all_hl_lists_ob, var).append(hl_list)
        sys.exit()
            

