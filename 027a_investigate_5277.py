from __future__ import division
import sys
import re

import numpy as np
import matplotlib.pyplot as plt

import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.myfilemanager as mfm
import LHCMeasurementTools.mystyle as ms

cell = '13L5'
fills = [5277, 5416, 5219]

re_variable = re.compile('^\w{5}_\d{2}[LR]\d_(\w{5})\.POSST$')

h5_file_raw ='/eos/user/l/lhcscrub/timber_data_h5/cryo_heat_load_data/cryo_data_fill_%i.h5'

plt.close('all')
myfontsz = 16
ms.mystyle_arial(fontsz=myfontsz, dist_tick_lab=8)
fig = plt.figure()
title = 'Cell %s' % cell
fig.canvas.set_window_title(title)
plt.suptitle(title, fontsize=16)
fig.patch.set_facecolor('w')
fig.subplots_adjust(left=.06, right=.88, top=.93, hspace=.38, wspace=.42)

for fill_ctr, filln in enumerate(fills):
    sp_ctr = fill_ctr % 4 + 1
    atd_ob = mfm.h5_to_obj(h5_file_raw % filln)
    tt = (atd_ob.timestamps - atd_ob.timestamps[0])/3600.
    sp = plt.subplot(2,2,sp_ctr)
    sp.set_title('Fill %i' % filln)

    color_ctr = 0
    for ctr, var in enumerate(atd_ob.variables):
        if cell in var:
            color_ctr += 1
            label = re_variable.search(var).group(1)
            sp.plot(tt, atd_ob.data[:,ctr], lw=2, label=label, color=ms.colorprog(color_ctr, 9))
    sp.legend(bbox_to_anchor=(1.15,1))
    ymin, ymax = sp.get_ylim()
    sp.set_ylim(ymin-1, ymax+1)
    plt.show()
