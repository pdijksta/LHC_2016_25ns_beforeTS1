import sys
import copy
import matplotlib
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pickle
import time

import LHCMeasurementTools.mystyle as ms
from LHCMeasurementTools.TimberManager import timb_timestamp2float_UTC
from LHCMeasurementTools.mystyle import colorprog
from hl_dicts.LHC_Heat_load_dict import mask_dict, main_dict
from LHCMeasurementTools.LHC_Fills import Fills_Info
import LHCMeasurementTools.TimestampHelpers as TH

moment = 'stable_beams'
moment = 'stop_squeeze'
#moment = 'start_ramp'
#moment = 'sb+2_hrs'

main_dict_0 = copy.deepcopy(main_dict)

mask = main_dict[moment]['n_bunches']['b1'] > 400
#~ mask = np.array(map(lambda n: n in [5219, 5222, 5223], main_dict['filln']))

main_dict = mask_dict(main_dict,mask)

# add timestamp
#merge all fill pickles

folders = ['./', '/afs/cern.ch/project/spsecloud/LHC_2015_PhysicsAfterTS2/', '/afs/cern.ch/project/spsecloud/LHC_2015_PhysicsAfterTS3/', 
            '/afs/cern.ch/project/spsecloud/LHC_2015_Scrubbing50ns/', '/afs/cern.ch/project/spsecloud/LHC_2015_IntRamp50ns/', 
            '/afs/cern.ch/project/spsecloud/LHC_2015_IntRamp25ns/']
dict_fill_bmodes = {}
for folder in folders:
    with open(folder+'fills_and_bmodes.pkl') as fid:   
        dict_fill_bmodes.update(pickle.load(fid))
        
for this_dict in [main_dict, main_dict_0]:
    this_dict['t_start_fill'] = []
    for filln in this_dict['filln']:
        this_dict['t_start_fill'].append(dict_fill_bmodes[filln]['t_startfill'])
    
    
    



fontsz = 14

plt.close('all')
ms.mystyle_arial(fontsz=fontsz, dist_tick_lab=5)

# intensity
fig1 = plt.figure(1, figsize = (8*1.5,6*1.5))
fig1.set_facecolor('w')


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

sphl1 = plt.subplot(2,1,1, sharex=sp3)
sphl2 = plt.subplot(2,1,2, sharex=sp3)


hl_keys.sort()
arc_ctr = 0
for key in hl_keys:
    if key[0] != 'S':
        continue
    
    color = colorprog(arc_ctr, 8)
    sphl1.plot(main_dict['filln'], main_dict[moment]['heat_load'][key], '.', label=key, color=color, markersize=12)
    sphl1.grid('on')
    
    sphl2.plot(main_dict['filln'], main_dict[moment]['heat_load'][key]/main_dict[moment]['intensity']['total'], '.', label=key, color=color, markersize=12)
    sphl2.grid('on')
    #~ sphl2.set_ylim(0, .4e-12)
    
    
    arc_ctr += 1

sphl2.legend(bbox_to_anchor=(1.1,1))
sphl1.set_ylabel('Heat load [W/hcell]')
sphl2.set_ylabel('Normalized heat load [W/hcell/p+]')   

for fig in [fig2]:
    fig.suptitle('At '+moment)
    
#chamonix  

mode = 'absolute'
mode = 'normalized'

tc = lambda t_stamps: matplotlib.dates.date2num(map(datetime.datetime.fromtimestamp, np.atleast_1d(t_stamps)))
pl = plt

figc = pl.figure(100, figsize=(6*2.4, 6))
figc.set_facecolor('w')

spc_hl16 = pl.subplot2grid((3,3),(1,1), rowspan=2, colspan=2)
spc_nbun16 = pl.subplot2grid((3,3),(0,1), rowspan=1, colspan=2, sharex=spc_hl16)

spc_hl15 = pl.subplot2grid((3,3),(1,0), rowspan=2, colspan=1, sharey=spc_hl16)
spc_nbun15 = pl.subplot2grid((3,3),(0,0), rowspan=1, colspan=1, sharex=spc_hl15, sharey=spc_nbun16)

for temp in [spc_nbun15, spc_nbun16]:
    temp.plot(tc(main_dict_0['t_start_fill']), main_dict_0[moment]['n_bunches']['b1'],'.b', markersize=12)


sector_list = ["S"+s for s in '12 23 34 45 56 67 78 81'.split()]

tc = lambda t_stamps: matplotlib.dates.date2num(map(datetime.datetime.fromtimestamp, np.atleast_1d(t_stamps)))

if mode == 'normalized':
    divide_by = main_dict[moment]['intensity']['total']
else:
    divide_by = 1.
    
#get_average  
hl_mat = []
int_hl_mat = []
for i_sec, sec in enumerate(sector_list):
    hl_mat.append(main_dict[moment]['heat_load'][sec]/divide_by)
    int_hl_mat.append( main_dict['integrated_hl'][sec])
avg_hl = np.mean(hl_mat, axis=0)
avg_int_hl = np.mean(int_hl_mat, axis=0)
    
for i_sec, sec in enumerate(sector_list):
    color = colorprog(i_sec, 8)
    for temp in [spc_hl15, spc_hl16]:
        temp.plot(tc(main_dict['t_start_fill']), main_dict[moment]['heat_load'][sec]/divide_by, '.', label=sec, color=color, markersize=12)

for temp in [spc_hl15, spc_hl16]:
    temp.plot(tc(main_dict['t_start_fill']), avg_hl, '.', label=sec, color='k', markersize=12)
    temp.plot(tc(main_dict['t_start_fill']), (main_dict[moment]['heat_load']['imp']['b1']+\
                                              main_dict[moment]['heat_load']['imp']['b2']+\
                                              main_dict[moment]['heat_load']['sr']['b1']+\
                                              main_dict[moment]['heat_load']['sr']['b2'])\
                                                *53.4/divide_by, '.', label=sec, color='grey', markersize=12)



#~ spc_nbun16.axes.get_yaxis().set_ticklabels([])
#~ spc_nbun16.axes.get_xaxis().set_ticklabels([])
#~ spc_nbun15.axes.get_xaxis().set_ticklabels([])
#~ spc_hl16.axes.get_yaxis().set_ticklabels([])


spc_hl16.set_xlim(tc(time.mktime(time.strptime('01 05 2016 00:00:00', '%d %m %Y %H:%M:%S')))[0],
                tc(time.mktime(time.strptime('15 11 2016 00:00:00', '%d %m %Y %H:%M:%S')))[0])
spc_hl15.set_xlim(tc(time.mktime(time.strptime('15 08 2015 00:00:00', '%d %m %Y %H:%M:%S')))[0],
                tc(time.mktime(time.strptime('15 11 2015 00:00:00', '%d %m %Y %H:%M:%S')))[0])

for ax in [spc_hl15, spc_hl16]:
    #~ hfmt = matplotlib.dates.DateFormatter('%a %d-%m %H:%M')
    hfmt = matplotlib.dates.DateFormatter('%b')
    ax.xaxis.set_major_formatter(hfmt)
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())

for ax in [spc_hl15, spc_hl16, spc_nbun15, spc_nbun16]:
    ax.grid('on')

if mode == 'normalized':
    spc_hl15.set_ylabel('Norm. heat load [W/hcell/p]')
else:
    spc_hl15.set_ylabel('Heat load [W/hcell]')

spc_nbun15.set_ylabel('N. bunches')
    
figc.subplots_adjust(hspace=.35)


# vs inteegrated dose
avg_int_hl[np.isnan(avg_int_hl)] = 0.

figi = pl.figure(101, figsize = (8*1.5,6*1.5))
figi.set_facecolor('w')
spi1 = pl.subplot2grid((4,1),(0,0))
pl.plot(np.cumsum(avg_int_hl), main_dict[moment]['n_bunches']['b1'], '.', color='b', markersize=12)
spi2 = pl.subplot2grid((4,1),(1,0), rowspan=3, sharex = spi1)


for i_sec, sec in enumerate(sector_list):
    color = colorprog(i_sec, 8)
    spi2.plot(np.cumsum(avg_int_hl), main_dict[moment]['heat_load'][sec]/divide_by, '.', label=sec, color=color, markersize=12)
pl.plot(np.cumsum(avg_int_hl), avg_hl, '.', color='k', markersize=12)

for ss in [spi1, spi2]:
    ss.grid('on')
    
spi2.set_xlabel('Integrated heat load [J/hcell]')
spi2.set_ylabel('Normalized heat load [W/hcell]')
spi1.set_ylabel('N. bunches')
spi2.set_xlim(0, 5.5e8)

figi.subplots_adjust(hspace=.35)

plt.show()
