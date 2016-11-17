import cPickle

import matplotlib.pyplot as plt
import numpy as np

from LHCMeasurementTools.TimberManager import timb_timestamp2float_UTC

dict_file = './large_heat_load_dict.pkl'

with open(dict_file, 'r') as f:
    main_dict = cPickle.load(f)

keys = main_dict.keys()
keys.sort()
print(keys)

plt.plot(main_dict['filln'], main_dict['tot_int_start_ramp'])
plt.show()

