# Rerun 024 script with same options
import cPickle
import os
import sys

dry_run = True

with open('./heatload_arcs.pkl.bck', 'r') as f:
    heatload_dict = cPickle.load(f)

fill_time_dict = {}
for key in sorted(heatload_dict.keys()):
    fill, time = key.split()
    if fill not in fill_time_dict:
        fill_time_dict[fill] = {'times': [], 'ap': heatload_dict[key]['Options']['Avg_period']}
        first_dict = heatload_dict[key]['Options']
    fill_time_dict[fill]['times'].append(time)
    this_dict = heatload_dict[key]['Options']
    for diff_key in 'Offset', 'Avg_period':
        if this_dict[diff_key] != first_dict[diff_key]:
            print fill, diff_key
            print this_dict[diff_key]
            print first_dict[diff_key]
            if diff_key == 'Avg_period':
                fill_time_dict[fill]['ap'] = min(fill_time_dict[fill]['ap'], this_dict[diff_key])

for fill, dd in fill_time_dict.items():
    time_arr = dd['times']
    times = ' '.join(time_arr)
    main_key = fill+' '+time_arr[0]
    avg_period = dd['ap']
    offset = heatload_dict[main_key]['Options']['Offset']
    exec_str = 'python 024_calc_arc_heat_load_fixed_time.py -fn -o %f %f -a %f %s %s' % (offset[0], offset[1], avg_period, fill, times)
    print('Running %s' % exec_str)
    if not dry_run:
        os.system(exec_str)
else:
    sys.exit()
