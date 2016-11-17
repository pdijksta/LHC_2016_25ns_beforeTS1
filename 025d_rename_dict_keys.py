import cPickle
import importlib
module025 = importlib.import_module('025_create_large_hl_dict')
output_key = module025.output_key


pkl_file = './large_heat_load_dict.pkl'

new_dict = {}

with open(pkl_file, 'r') as f:
    output_dict = cPickle.load(f)

for key in output_dict.keys():
    try:
        new_key = output_key(key)
    except ValueError:
        new_key = key
    new_dict[new_key] = output_dict[key]

with open(pkl_file, 'w') as f:
    cPickle.dump(new_dict, f, -1)
