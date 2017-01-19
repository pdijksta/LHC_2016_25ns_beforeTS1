import sys
import cPickle
import copy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import LHCMeasurementTools.mystyle as ms
from LHCMeasurementTools.TimberManager import timb_timestamp2float_UTC
from LHCMeasurementTools.mystyle import colorprog
import LHCMeasurementTools.TimestampHelpers as TH
from hl_dicts.LHC_Heat_load_dict import mask_dict, main_dict
#moment = 'start_ramp'
moment = 'stop_squeeze'
#moment = 'stable_beams'
#moment = 'sb+2_hrs'

date_on_xaxis = False
only_average = False #True

def mask_dict(dictionary, mask):
    new_dict = copy.deepcopy(dictionary)
    _mask_recursively(new_dict,mask)
    return new_dict

def _mask_recursively(dictionary, mask):
    for key in dictionary:
        if type(dictionary[key]) is dict:
            _mask_recursively(dictionary[key],mask)
        else:
            dictionary[key] = dictionary[key][mask]
filln_range = None

fontsz = 16

plt.close('all')
ms.mystyle_arial(fontsz=fontsz)#, dist_tick_lab=10)
fig1 = plt.figure(1, figsize = (8*1.5,6*1.5))
#fig1 = plt.figure(1, figsize = (8*0.95,6*1.5))
fig1.set_facecolor('w')

mask = np.logical_and(main_dict[moment]['n_bunches']['b1'] > 800, main_dict[moment]['n_bunches']['b2'] > 800)
#~ mask = np.array(map(lambda n: n in [5219, 5222, 5223], main_dict['filln']))

main_dict = mask_dict(main_dict,mask)

#mask = main_dict['bpi'] == 96
#main_dict = mask_dict(main_dict,mask)


# mask fill nrs
if filln_range is not None:
    mask = np.logical_and(main_dict['filln'] > filln_range[0], main_dict['filln'] < filln_range[1])
    main_dict = mask_dict(main_dict,mask)

if date_on_xaxis:
    time_in = 'datetime'
    t_plot_tick_h = None #'4weeks'
    time_conv = TH.TimeConverter(time_in, t_plot_tick_h=t_plot_tick_h)
    tc = time_conv.from_unix
    x_axis = tc(main_dict[moment]['t_stamps'])
else:
    x_axis = main_dict['filln']

sp1 = plt.subplot(4,1,1)
sp1.plot(x_axis, main_dict[moment]['n_bunches']['b1'],'.', markersize=12)
sp1.plot(x_axis, main_dict[moment]['n_bunches']['b2'],'r.', markersize=12)
sp1.set_ylabel('N bunches')
sp1.set_ylim(800, 2400)

sp2 = plt.subplot(4,1,2,sharex=sp1)
sp2.plot(x_axis, main_dict['bpi'],'.', markersize=12)
sp2.set_ylabel('Bpi')
sp2.set_ylim(60, 150)

sp3 = plt.subplot(4,1,3, sharex=sp1)
sp3.plot(x_axis, main_dict[moment]['intensity']['b1']/main_dict[moment]['n_bunches']['b1'],'b.', markersize=12)
sp3.plot(x_axis, np.array(main_dict[moment]['intensity']['b2'])\
/np.array(main_dict[moment]['n_bunches']['b2']),'r.', markersize=12)
sp3.set_ylabel('Bunch Intensity')
sp3.set_ylim(0.6e11, 1.3e11)

sp4 = plt.subplot(4,1,4, sharex=sp1)
sp4.plot(x_axis, main_dict[moment]['blength']['b1']['avg'],'b.', markersize=12)
sp4.plot(x_axis, main_dict[moment]['blength']['b2']['avg'],'r.', markersize=12)
sp4.set_ylabel('Bunch length')
sp4.set_ylim(1e-9, 1.4e-9)

if date_on_xaxis:
        time_conv.set_x_for_plot(fig1, sp1)
else:
    sp4.set_xlabel('Fill nr')
    sp4.xaxis.set_major_locator(ticker.MultipleLocator(100))

for sp in (sp1,sp2,sp3):
    plt.setp(sp.get_xticklabels(), visible = False)

sp1.grid('on')
sp2.grid('on')
sp3.grid('on')
sp4.grid('on')


hl_keys = main_dict[moment]['heat_load'].keys()

#sp_ctr = 0
fig2 = plt.figure(2, figsize = (8*1.5,6*1.5))
#fig2 = plt.figure(2, figsize = (8*0.95,6*1.5))
#fig2 = plt.figure(2, figsize = (8*0.95,6*1.))
fig2.set_facecolor('w')

sp5 = plt.subplot(2,1,1, sharex=sp1)
sp6 = plt.subplot(2,1,2, sharex=sp1)

fig3 = plt.figure(3, figsize = (8*1.5,6*1.5))
#fig3 = plt.figure(3, figsize = (8*.95,6*1.5))
fig3.set_facecolor('w')

sp7 = plt.subplot(1,1,1)
hl_keys.sort()
arc_ctr = 0
zeros = []
average = main_dict[moment]['heat_load']['S12']*0

for key in hl_keys:
    if key[0] != 'S':
        continue
    if only_average:
        average += main_dict[moment]['heat_load'][key]/8.
        continue
    color = colorprog(arc_ctr, 8)
    sp5.plot(x_axis, main_dict[moment]['heat_load'][key], '.', label=key, color=color, markersize=12)
    sp5.grid('on')
    sp5.set_ylim(0,180)
    
    sp6.plot(x_axis, main_dict[moment]['heat_load'][key]/main_dict[moment]['intensity']['total'], '.', label=key, color=color, markersize=12)
#    sp6.plot(x_axis, main_dict[moment]['heat_load'][key]/(main_dict[moment]['n_bunches']['b1']+main_dict[moment]['n_bunches']['b2'])#/(np.array(main_dict[moment]['intensity']['b1'])/np.array(main_dict[moment]['n_bunches']['b1'])+np.array(main_dict[moment]['intensity']['b2'])/np.array(main_dict[moment]['n_bunches']['b2'])/2)**0.2\
#, '.', label=key, color=color, markersize=12)
    sp6.grid('on')
    sp6.set_ylim(0, 5e-13)
    
    xx = main_dict[moment]['intensity']['total']/(main_dict[moment]['n_bunches']['b1']+main_dict[moment]['n_bunches']['b2'])/1e11
    yy = main_dict[moment]['heat_load'][key] - main_dict[moment]['heat_load']['total_model']*53.45
    fit = np.polyfit(xx,yy,1)
    yy_fit = np.poly1d(fit)
    xx_fit = np.arange(0.4, 2.5, 0.1)
    zeros.append(-fit[1]/fit[0])

    sp7.plot(xx, yy, '.', label=key, color=color, markersize=12)
    #sp7.plot(xx, main_dict[moment]['heat_load']['total_model']*53.45, '.', ls='--', color=color, markersize=12)
    sp7.plot(xx_fit, yy_fit(xx_fit), color=color, lw=3)
    if arc_ctr == 0:
        sp7.axhline(160, color='black', lw=3)
    sp7.grid('on')
    sp7.set_ylim(0,350)
    sp7.set_xlim(0,2.5)
    
    arc_ctr += 1

if only_average:
    sp5.plot(x_axis, average, 'k.', markersize=12)
    sp5.grid('on')
    sp5.set_ylim(0,120)
    
    sp6.plot(x_axis, average/main_dict[moment]['intensity']['total'], 'k.', markersize=12)
    sp6.grid('on')
    sp6.set_ylim(0, 4e-13)
    

#sp6.legend(bbox_to_anchor=(1.22,1.04))
sp6.legend(bbox_to_anchor=(1.50,1.04))
sp5.set_ylabel('Heat load [W/hcell]')
sp6.set_ylabel('Normalized heat load [W/hcell/p+]')   
sp7.set_ylabel('Heat load [W/hcell]')
sp7.set_xlabel('Bunch intensity [1e11]')   
plt.setp(sp5.get_xticklabels(), visible = False)

if date_on_xaxis:
    time_conv.set_x_for_plot(fig2, sp1)
else:
    sp6.set_xlabel('Fill nr')

for fig in [fig1, fig2, fig3]:
    fig.suptitle('At '+moment)
    #fig.subplots_adjust(right=0.83)
    fig.subplots_adjust(right=0.7, left=0.15)

#fig1.savefig('overview_%s_%s'%(year, moment), dpi = 200)
#fig2.savefig('overview_%s_%s_hl'%(year, moment), dpi = 200)
#fig1.savefig('overview_%s_%s_scaled3'%(year, moment), dpi = 200)
#fig2.savefig('overview_%s_%s_hl_scaled3'%(year, moment), dpi = 200)
#fig3.savefig('overview_%s_%s_fit'%(year, moment), dpi = 200)

plt.show()
