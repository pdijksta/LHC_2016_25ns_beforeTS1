import numpy as np
import matplotlib.pyplot as plt
import re

from hl_dicts.LHC_Heat_load_dict import main_dict
from LHCMeasurementTools.mystyle import colorprog

plt.rcParams['lines.markersize'] = 10
plt.close('all')

re_list = ['^S\d\d$', '^Q6[RL]\d$']
title_list = ['Arcs', 'Q6 Quads']
ylim_list = [(0,1.5e-12), (0,3.5e-13)]

int_dict = main_dict['integrated_hl']
sp = None
fig = plt.figure(figsize = (8*1.5,6*1.5))
fig.set_facecolor('w')
fig.canvas.set_window_title('Integrated heat load')

for ctr, (reg, title, ylim) in enumerate(zip(re_list, title_list, ylim_list)):
    regex = re.compile(reg)
    sp = plt.subplot(2,1,ctr+1, sharex=sp)
    sp.set_title('Integrated heat load')
    sp.set_ylabel('Cumulated HL [J]')
    sp.set_title(title)
    sp.grid(True)
    
    sp2 = sp.twinx()
    sp2.set_ylabel('Normalized HL [W/p+]')
    sp2.set_ylim(*ylim)
    good_keys = filter(regex.match, int_dict.keys())
    for key_ctr, key in enumerate(good_keys):
        item = int_dict[key]
        color = colorprog(key_ctr,8)
        sp.plot(main_dict['filln'], np.cumsum(item), label=key, color=color)
        sp2.plot(main_dict['filln'], \
                main_dict['stable_beams']['heat_load'][key]/main_dict['stable_beams']['intensity']['total'],\
                '.', color=color)
    sp.legend(bbox_to_anchor=(1.15,1))
sp.set_xlabel('Fill #')

fig = plt.figure(figsize = (8*1.5,6*1.5))
fig.set_facecolor('w')
fig.canvas.set_window_title('Integrated heat load 2')

ylim_list = [(0,1e-12), (0, .5e-12)]
for ctr, (reg, title, ylim) in enumerate(zip(re_list, title_list, ylim_list)):
    regex = re.compile(reg)
    sp = plt.subplot(2,1,ctr+1)
    sp.set_title('Integrated heat load')
    sp.set_ylabel('Normalized HL [W/p+]')
    sp.set_title(title)
    sp.grid(True)
    
    good_keys = filter(regex.match, int_dict.keys())
    for key_ctr, key in enumerate(good_keys):
        item = int_dict[key]
        color = colorprog(key_ctr,8)
        norm_hl = main_dict['stable_beams']['heat_load'][key]/main_dict['stable_beams']['intensity']['total']
        sp.plot(np.cumsum(item), norm_hl,'.', label=key, color=color)
    sp.legend(bbox_to_anchor=(1.15,1))
    sp.set_ylim(*ylim)
    sp.set_xlim(0,None)
sp.set_xlabel('Cumulated HL [J]')

fig.subplots_adjust(left=.06, right=.84, top=.93, hspace=.38, wspace=.42)
plt.show()
