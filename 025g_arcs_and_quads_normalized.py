import sys
import cPickle
import copy
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import LHCMeasurementTools.mystyle as ms
from LHCMeasurementTools.TimberManager import timb_timestamp2float_UTC
from LHCMeasurementTools.mystyle import colorprog
import LHCMeasurementTools.TimestampHelpers as TH
import LHCMeasurementTools.LHC_Heatloads as hl
from hl_dicts.LHC_Heat_load_dict import mask_dict, main_dict, arc_list

plt.close('all')
#moment = 'start_ramp'
moment = 'stop_squeeze'
#moment = 'stable_beams'
#moment = 'sb+2_hrs'

date_on_xaxis = True
filln_range = None # Tuple of min / max fill

fontsz = 16
ms.mystyle_arial(fontsz=fontsz)#, dist_tick_lab=10)

mask = np.logical_and(main_dict[moment]['n_bunches']['b1'] > 800, main_dict[moment]['n_bunches']['b2'] > 800)
main_dict = mask_dict(main_dict,mask)

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

fig1 = plt.figure(1)
fig1.set_facecolor('w')

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

fig2 = plt.figure(2)
fig2.set_facecolor('w')

sp5 = plt.subplot(3,1,1, sharex=sp1)
sp5.grid('on')
sp5.set_ylim(0,180)

sp6 = plt.subplot(3,1,2, sharex=sp1)
sp6.grid('on')
sp6.set_ylim(0, 5e-13)

sp_avg = plt.subplot(3,1,3, sharex=sp1)
sp_avg.grid('on')

sp5.set_title('Arc heat loads')
sp6.set_title('Normalized arc heat loads.')
sp_avg.set_title('Normalized arc heat loads - Delta to average arc')

fig3 = plt.figure(3)
fig3.set_facecolor('w')

sp7 = plt.subplot(1,1,1)
sp7.grid('on')
sp7.set_ylim(0,350)
sp7.set_xlim(0,2.5)

zeros = []

average = 0
for key in arc_list:
    average += main_dict[moment]['heat_load'][key]/len(arc_list)

tot_int = main_dict[moment]['intensity']['total']
sp5.plot(x_axis, average, '.', label='Average', color='black', markersize=12)
sp6.plot(x_axis, average/tot_int, '.', label='Average', color='black', markersize=12)
sp_avg.plot(x_axis, np.zeros_like(average), '.', label='Average', color='black', markersize=12)

for arc_ctr, key in enumerate(arc_list):
    color = colorprog(arc_ctr, 8)
    arc_hl = main_dict[moment]['heat_load'][key]

    sp5.plot(x_axis, arc_hl, '.', label=key, color=color, markersize=12)
    sp6.plot(x_axis, arc_hl/tot_int, '.', label=key, color=color, markersize=12)
    sp_avg.plot(x_axis, (arc_hl-average)/tot_int, '.', label=key, color=color, markersize=12)
    
    xx = tot_int/(main_dict[moment]['n_bunches']['b1']+main_dict[moment]['n_bunches']['b2'])/1e11
    yy = arc_hl - main_dict[moment]['heat_load']['total_model']*53.45
    fit = np.polyfit(xx,yy,1)
    yy_fit = np.poly1d(fit)
    xx_fit = np.arange(0.4, 2.5, 0.1)
    zeros.append(-fit[1]/fit[0])

    sp7.plot(xx, yy, '.', label=key, color=color, markersize=12)
    #sp7.plot(xx, main_dict[moment]['heat_load']['total_model']*53.45, '.', ls='--', color=color, markersize=12)
    sp7.plot(xx_fit, yy_fit(xx_fit), color=color, lw=3)
    if arc_ctr == 0:
        sp7.axhline(160, color='black', lw=3)


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

# Quads
hl_keys = main_dict[moment]['heat_load'].keys()
re_q6 = re.compile('^Q6')
quad_keys = filter(re_q6.match, hl_keys)
def get_len(q6_str):
    q6_nr = int(q6_str[-1])
    return hl.magnet_length['Q6s_IR%i' % q6_nr][0]
quad_lens = map(get_len, quad_keys)

fig4 = plt.figure()
fig4.set_facecolor('w')

sp5 = plt.subplot(3,1,1, sharex=sp1)
sp6 = plt.subplot(3,1,2, sharex=sp1)
sp_avg = plt.subplot(3,1,3, sharex=sp1)

sp5.set_title('Quad heat loads')
sp6.set_title('Normalized quad heat loads.')
sp_avg.set_title('Normalized quad heat loads - Delta to average arc')

sp5.set_ylabel('Quad heat loads [W]')
sp6.set_ylabel('Quad heat loads [W/p+]')
sp_avg.set_ylabel('Quad heat loads [W/p+]')


for sp in sp5, sp6, sp_avg:
    sp.grid(True)

average = 0
for key, len_ in zip(quad_keys, quad_lens):
    average += main_dict[moment]['heat_load'][key]/len_/len(quad_keys)

sp5.plot(x_axis, average, '.', label='Average', color='black', markersize=12)
sp6.plot(x_axis, average/tot_int, '.', label='Average', color='black', markersize=12)
sp_avg.plot(x_axis, np.zeros_like(average), '.', label='Average', color='black', markersize=12)

for ctr, (key, len_) in enumerate(zip(quad_keys, quad_lens)):
    color = colorprog(ctr, quad_keys)
    this_hl = main_dict[moment]['heat_load'][key]/len_

    sp5.plot(x_axis, this_hl, '.', label=key, color=color, markersize=12)
    sp6.plot(x_axis, this_hl/tot_int, '.', label=key, color=color, markersize=12)
    sp_avg.plot(x_axis, (this_hl-average)/tot_int, '.', label=key, color=color, markersize=12)
    
#    xx = tot_int/(main_dict[moment]['n_bunches']['b1']+main_dict[moment]['n_bunches']['b2'])/1e11
#    yy = this_hl - main_dict[moment]['heat_load']['total_model']*53.45
#    fit = np.polyfit(xx,yy,1)
#    yy_fit = np.poly1d(fit)
#    xx_fit = np.arange(0.4, 2.5, 0.1)
#    zeros.append(-fit[1]/fit[0])
#
#    sp7.plot(xx, yy, '.', label=key, color=color, markersize=12)
#    #sp7.plot(xx, main_dict[moment]['heat_load']['total_model']*53.45, '.', ls='--', color=color, markersize=12)
#    sp7.plot(xx_fit, yy_fit(xx_fit), color=color, lw=3)

plt.setp(sp5.get_xticklabels(), visible = False)
plt.setp(sp6.get_xticklabels(), visible = False)
sp6.legend(bbox_to_anchor=(1.50,1.04))


for fig in [fig1, fig2, fig3, fig4]:
    fig.suptitle('At '+moment)
    #fig.subplots_adjust(right=0.83)
    fig.subplots_adjust(right=0.7, left=0.15)

#fig1.savefig('overview_%s_%s'%(year, moment), dpi = 200)
#fig2.savefig('overview_%s_%s_hl'%(year, moment), dpi = 200)
#fig1.savefig('overview_%s_%s_scaled3'%(year, moment), dpi = 200)
#fig2.savefig('overview_%s_%s_hl_scaled3'%(year, moment), dpi = 200)
#fig3.savefig('overview_%s_%s_fit'%(year, moment), dpi = 200)

plt.show()
