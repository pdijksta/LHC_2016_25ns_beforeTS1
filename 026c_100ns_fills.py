from __future__ import division
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt

import hl_dicts.LHC_Heat_load_dict as hld
import LHCMeasurementTools.LHC_Heatloads as HL
import LHCMeasurementTools.mystyle as ms
import LHCMeasurementTools.savefig as sf

from GasFlowHLCalculator.config_qbs import arc_list

parser = argparse.ArgumentParser()
parser.add_argument('--pdsave', help='Save in pdijksta dir', action='store_true')
parser.add_argument('--noshow', help='Do not call plt.show', action='store_true')
parser.add_argument('--nosubtract', help='Do not subtract offsets for bar plots', action='store_true')
args = parser.parse_args()

arc_length = HL.magnet_length['AVG_ARC']
moment = 'stable_beams'
hl_dict = hld.hl_dict

plt.close('all')
ms.mystyle_arial()
plt.rcParams['lines.markersize'] = 9

re_100ns = re.compile('^100ns_')
mask = []
for fp in hl_dict['filling_pattern']:
    mask.append(re_100ns.match(fp) != None)

dict_100ns = hld.mask_dict(hl_dict, np.array(mask))
fills = dict_100ns['filln']
print('Fills with 100ns: %s' % fills)

# Gain information on fills
fig = plt.figure()
title = '100 ns fills at stable beams'
fig.canvas.set_window_title(title)
fig.patch.set_facecolor('w')
fig.subplots_adjust(left=.06, right=.84, top=.93, hspace=.38, wspace=.42)
plt.suptitle(title, fontsize=20)

# Intensity
sp = plt.subplot(2,2,1)
sp.set_title('Intensity and Energy')
sp.set_ylabel('Intensity [p+]')
sp.set_xlabel('Fill #')
sp2 = sp.twinx()
sp2.set_ylabel('Energy [TeV]')

sp2.plot(fills, dict_100ns['energy']/1e12, '.', label='Energy', color='red')
sp.plot(fills, dict_100ns[moment]['intensity']['b1'],'.', label='Int B1')
sp.plot(fills, dict_100ns[moment]['intensity']['b2'],'.', label='Int B2')
ms.comb_legend(sp, sp2, bbox_to_anchor=(1.25,1))

# N bunches and B int
sp = plt.subplot(2,2,2)
sp.set_title('Bunches')
sp.set_ylabel('# of bunches')
sp.set_xlabel('Fill #')
sp2 = sp.twinx()
sp2.set_ylabel('Bunch Intensity')
nb1 = dict_100ns[moment]['n_bunches']['b1']
nb2 = dict_100ns[moment]['n_bunches']['b2']
sp.plot(fills, nb1, '.', label='# bunches B1')
sp.plot(fills, nb2, '.', label='# bunches B2')
sp2.plot(fills, dict_100ns[moment]['intensity']['b1']/nb1,'.', label='Bunch int B1', marker='x')
sp2.plot(fills, dict_100ns[moment]['intensity']['b2']/nb2,'.', label='Bunch int B2', marker='x')
ms.comb_legend(sp, sp2, bbox_to_anchor=(1.4,1))

# Intensity over time
sp = plt.subplot(2,2,3)
sp.set_title('Intensity over time')
sp.set_ylabel('Intensity')
sp.set_xlabel('Time after sb [h]')
sp.grid(True)

tot_int = hld.values_over_time(dict_100ns, 'intensity', 'total')

for fill_ctr, fill in enumerate(fills):
    xy = tot_int[fill_ctr]
    sp.plot(xy[0], xy[1],'.-', label=fill)
sp.legend(bbox_to_anchor=(1.2,1))

# Heat loads

# For each Arc
model_hl = hld.values_over_time(dict_100ns, 'heat_load', 'total_model')
for arc_ctr, arc in enumerate(arc_list):
    sp_ctr = arc_ctr % 4 + 1
    if sp_ctr ==1:
        fig = plt.figure()
        title = '100 ns fills at stable beams - Heat loads'
        fig.canvas.set_window_title(title)
        fig.patch.set_facecolor('w')
        fig.subplots_adjust(left=.06, right=.84, top=.93, hspace=.38, wspace=.42)
        plt.suptitle(title, fontsize=20)
    sp = plt.subplot(2,2,sp_ctr)
    sp.set_title('Arc %s' % arc)
    sp.set_ylabel('Heat load per half cell [W]')
    sp.set_xlabel('Time after sb [h]')
    sp.grid(True)
    hl = hld.values_over_time(dict_100ns, 'heat_load', 'arc_averages', arc)
    for ctr, (hl_arr, fill, modelz) in enumerate(zip(hl, fills, model_hl)):
        color = ms.colorprog(ctr, fills)
        sp.plot(hl_arr[0],hl_arr[1],'.-', label=fill, color=color)
        if ctr == 0:
            label = 'Subtracted model hl'
        else:
            label = None
        sp.plot(modelz[0], hl_arr[1]-modelz[1]*arc_length, '.--', label=label, color=color)
    sp.legend(bbox_to_anchor=(1.2,1))

# For each fill
for ctr, (fill, modelz) in enumerate(zip(fills, model_hl)):
    sp_ctr = ctr % 4 + 1
    if sp_ctr ==1:
        fig = plt.figure()
        title = '100 ns fills at stable beams - Heat loads'
        fig.canvas.set_window_title(title)
        fig.patch.set_facecolor('w')
        fig.subplots_adjust(left=.06, right=.84, top=.93, hspace=.38, wspace=.42)
        plt.suptitle(title, fontsize=20)
    sp = plt.subplot(2,2,sp_ctr)
    sp.set_title('Fill %i' % fill)
    sp.set_ylabel('Heat load per half cell [W]')
    sp.set_xlabel('Time after sb [h]')
    sp.grid(True)
    for arc_ctr, arc in enumerate(arc_list):
        hl_arr = hld.values_over_time(dict_100ns, 'heat_load', 'arc_averages', arc)[ctr]
        color = ms.colorprog(arc_ctr, arc_list)
        sp.plot(hl_arr[0],hl_arr[1],'.-', label=arc, color=color)
        if ctr == 0:
            label = 'Subtracted model hl'
        else:
            label = None
        sp.plot(modelz[0], hl_arr[1]-modelz[1]*arc_length, '.--', label=label, color=color)
    if sp_ctr == 2:
        sp.legend(bbox_to_anchor=(1.2,1))

# Cell by cell heat load
index = -1
filln = fills[index]

for arc_ctr, arc in enumerate(arc_list):
    arc_str = 'Arc_'+arc[-2:]
    sp_ctr = arc_ctr % 2 + 1
    data_dict = dict_100ns[moment]['heat_load']['all_cells']
    if sp_ctr == 1:
        fig = plt.figure(figsize=(12,8))
        title = 'Fill %i at %s.' % (filln, moment)
        if args.nosubtract:
            title += ' Offsets not subtracted'
        else:
            title += ' Offsets subtracted'
        fig.canvas.set_window_title(title)
        fig.patch.set_facecolor('w')
        fig.subplots_adjust(left=.1, right=.75, top=.87, hspace=.45, wspace=.45)
        plt.suptitle(title, fontsize=18)
    sp = plt.subplot(2,1,sp_ctr)
    sp.set_ylabel('Heat load [W/hc]')
    sp.set_xlabel('Cell')
    sp.set_title(arc)
    sp.grid(True)
    for ctr, cell in enumerate(hld.arc_cells_dict[arc_str]):
        if args.nosubtract:
            yy = data_dict[cell][index] + dict_100ns['hl_subtracted_offset']['all_cells'][cell][index]
        else:
            yy = data_dict[cell][index]
        sp.bar(ctr, yy)
    sp.axhline(dict_100ns[moment]['heat_load']['imp']['total'][index]*53.45, color='orange', label='Impedance')
    sp.axhline(dict_100ns[moment]['heat_load']['total_model'][index]*53.45, color='green', label='Impedance + SR')
    if args.nosubtract:
        arc_average = dict_100ns[moment]['heat_load']['arc_averages'][arc][index]+dict_100ns['hl_subtracted_offset']['arc_averages'][arc][index]
    else:
        arc_average = dict_100ns[moment]['heat_load']['arc_averages'][arc][index]
        sp.axhline(dict_100ns['hl_subtracted_offset']['arc_averages'][arc][index], color='red', label='Average subtracted offset')
    sp.axhline(arc_average, color='blue', label='Arc Average')

    sp.legend(bbox_to_anchor=(1.4,1))

if args.pdsave:
    sf.saveall_pdijksta()

if not args.noshow:
    plt.show()
