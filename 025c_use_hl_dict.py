import sys
import copy

import matplotlib.pyplot as plt
import numpy as np

import LHCMeasurementTools.mystyle as ms
from LHCMeasurementTools.TimberManager import timb_timestamp2float_UTC
from LHCMeasurementTools.mystyle import colorprog
from hl_dicts.LHC_Heat_load_dict import mask_dict, main_dict

moment = 'stable_beams'
moment = 'stop_squeeze'
#moment = 'start_ramp'
#moment = 'sb+2_hrs'

main_dict_0 = copy.deepcopy(main_dict)

fontsz = 16

plt.close('all')
ms.mystyle_arial(fontsz=fontsz, dist_tick_lab=10)

# intensity
fig1 = plt.figure(1, figsize = (8*1.5,6*1.5))
fig1.set_facecolor('w')

mask = main_dict[moment]['n_bunches']['b1'] > 1500
#~ mask = np.array(map(lambda n: n in [5219, 5222, 5223], main_dict['filln']))

main_dict = mask_dict(main_dict,mask)

sp1 = plt.subplot(4,1,1)
sp1.plot(main_dict['filln'], main_dict[moment]['n_bunches']['b1'],'.', markersize=12)
sp1.set_ylabel('N bunches')

sp2 = plt.subplot(4,1,2,sharex=sp1)
sp2.plot(main_dict['filln'], main_dict['bpi'],'.', markersize=12)
sp2.set_ylabel('Bpi')

sp3 = plt.subplot(4,1,3, sharex=sp1)
sp3.plot(main_dict['filln'], main_dict[moment]['intensity']['b1']/main_dict[moment]['n_bunches']['b1'],'b.', markersize=12)
sp3.plot(main_dict['filln'], np.array(main_dict[moment]['intensity']['b2'])\
/np.array(main_dict[moment]['n_bunches']['b2']),'r.', markersize=12)
sp3.set_ylabel('Bunch Intensity')

sp4 = plt.subplot(4,1,4, sharex=sp1)
sp4.plot(main_dict['filln'], main_dict[moment]['blength']['b1']['avg'],'b.', markersize=12)
sp4.plot(main_dict['filln'], main_dict[moment]['blength']['b2']['avg'],'r.', markersize=12)
sp4.set_ylabel('Bunch length')

sp1.grid('on')
sp2.grid('on')
sp3.grid('on')
sp4.grid('on')


fig1.suptitle('At '+moment)
hl_keys = main_dict[moment]['heat_load'].keys()

# Heat load
fig2 = plt.figure(2, figsize = (8*1.5,6*1.5))
fig2.set_facecolor('w')

sp1 = plt.subplot(2,1,1, sharex=sp3)
sp2 = plt.subplot(2,1,2, sharex=sp3)

fig3 = plt.figure(3, figsize = (8*1.5,6*1.5))
fig3.set_facecolor('w')

sp3 = plt.subplot(1,1,1)
hl_keys.sort()
arc_ctr = 0
for key in hl_keys:
    if key[0] != 'S':
        continue
    
    color = colorprog(arc_ctr, 8)
    sp1.plot(main_dict['filln'], main_dict[moment]['heat_load'][key], '.', label=key, color=color, markersize=12)
    sp1.grid('on')
    
    sp2.plot(main_dict['filln'], main_dict[moment]['heat_load'][key]/main_dict[moment]['intensity']['total'], '.', label=key, color=color, markersize=12)
    sp2.grid('on')
    sp2.set_ylim(0, .4e-12)
    
    xx = main_dict[moment]['intensity']['total']/main_dict[moment]['n_bunches']['b1']/2.
    yy = main_dict[moment]['heat_load'][key] - main_dict[moment]['heat_load']['total_model']*53.45
    fit = np.polyfit(xx,yy,1)
    yy_fit = np.poly1d(fit)
    xx_fit = np.arange(0.4e11, 2.5e11, 0.1e11)
    
    sp3.plot(xx, yy, '.', label=key, color=color, markersize=12)
    #sp3.plot(xx, main_dict[moment]['heat_load']['total_model']*53.45, '.', ls='--', color=color, markersize=12)
    sp3.plot(xx_fit, yy_fit(xx_fit), color=color, lw=3)
    if arc_ctr == 0:
        sp3.axhline(160, color='black', lw=3)
    sp3.grid('on')
    sp3.set_ylim(0,350)
    sp3.set_xlim(0,2.5e11)
    
    arc_ctr += 1

sp2.legend(bbox_to_anchor=(1.1,1))
sp1.set_ylabel('Heat load [W/hcell]')
sp2.set_ylabel('Normalized heat load [W/hcell/p+]')   

for fig in [fig2, fig3]:
    fig.suptitle('At '+moment)

# Time between end of ramp and stable beams / end of squeeze
t_stop_ramp = main_dict_0['start_ramp']['t_stamps'] + 1223
t_sb = main_dict_0['stable_beams']['t_stamps']
t_stop_squeeze = main_dict_0['stop_squeeze']['t_stamps']

mask = np.logical_and(t_sb > 0, t_sb-t_stop_ramp < 100*60)
mask = np.logical_and(mask, t_stop_squeeze > t_stop_ramp)
mask = np.logical_and(mask, main_dict_0['energy'] > 6.4e12)
fig = plt.figure()
fig.set_facecolor('w')
sp = plt.subplot(1,1,1)
title = 'Time needed after ramp'
fig.canvas.set_window_title(title)

fig.suptitle(title, fontsize=25)
sp.set_xlabel('Fill number')
sp.set_ylabel('Time [min]')

sp.plot(main_dict_0['filln'][mask], (t_stop_squeeze[mask] - t_stop_ramp[mask])/60., '.', label='Stop squeeze', markersize=12)
sp.plot(main_dict_0['filln'][mask], (t_sb[mask] - t_stop_ramp[mask])/60., '.', label='Stable beams', markersize=12)
sp.legend(bbox_to_anchor=(1.1,1))

plt.show()
