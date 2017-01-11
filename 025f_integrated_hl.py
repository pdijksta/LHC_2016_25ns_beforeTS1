import numpy as np
import matplotlib.pyplot as plt
import re

from LHCMeasurementTools.LHC_Heat_load_dict import main_dict
from LHCMeasurementTools.mystyle import colorprog

plt.close('all')
fig = plt.figure(figsize = (8*1.5,6*1.5))
fig.set_facecolor('w')
fig.canvas.set_window_title('Integrated heat load')

re_list = ['^S\d\d$', '^Q6[RL]\d$']
title_list = ['Arcs', 'Q6 Quads']
ylim_list = [(0,5e-13), (0,3.5e-13)]

int_dict = main_dict['integrated_hl']
sp = None
for ctr, (reg, title, ylim) in enumerate(zip(re_list, title_list, ylim_list)):
    regex = re.compile(reg)
    sp = plt.subplot(3,1,ctr+1, sharex=sp)
    sp.set_title('Integrated heat load')
    sp.set_ylabel('Cumulated HL [J]')
    sp.set_title(title)
    sp.grid(True)
    
    sp2 = sp.twinx()
    sp2.set_ylabel('Heat load normalized by intensity')
    sp2.set_ylim(*ylim)
    for key_ctr, (key, item) in enumerate(int_dict.iteritems()):
        if regex.match(key):
            color = colorprog(key_ctr,8)
            sp.plot(main_dict['filln'], np.cumsum(item), label=key, color=color)
            sp2.plot(main_dict['filln'], \
                    main_dict['stable_beams']['heat_load'][key]/main_dict['stable_beams']['intensity']['total'],\
                    '.', color=color)
    sp.legend(bbox_to_anchor=(1.15,1))
sp.set_xlabel('Fill #')

fig.subplots_adjust(left=.06, right=.84, top=.93, hspace=.38, wspace=.42)
plt.show()
