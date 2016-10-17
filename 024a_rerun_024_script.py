# Rerun 024 script with same options
import cPickle
import os

with open('./heatload_arcs.pkl', 'r') as f:
    heatload_dict = cPickle.load(f)

for main_key in heatload_dict:
    avg_period = heatload_dict[main_key]['Options']['Avg_period']
    offset = heatload_dict[main_key]['Options']['Offset']
    exec_str = 'python 024_calc_arc_heat_load_fixed_time.py -f -o %f %f -a %f %s' %  (offset[0], offset[1], avg_period, main_key)
    print('Running %s' % exec_str)
    os.system(exec_str)

