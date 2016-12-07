import sys
import cPickle
import copy

import matplotlib.pyplot as plt
import numpy as np

import LHCMeasurementTools.mystyle as ms
from LHCMeasurementTools.TimberManager import timb_timestamp2float_UTC
from LHCMeasurementTools.mystyle import colorprog

dict_file_2016 = './large_heat_load_dict_2016.pkl'
dict_file_2015 = './large_heat_load_dict_2015.pkl'

moment = 'stable_beams'
#moment = 'start_ramp'
#moment = 'sb+2_hrs'

with open(dict_file_2016, 'r') as f:
    main_dict_2016 = cPickle.load(f)
with open(dict_file_2015, 'r') as f:
    main_dict_2015 = cPickle.load(f)

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

def merge_dicts(dict1,dict2):
    new_dict = copy.deepcopy(dict1)
    _merge_dicts_recursively(dict1,dict2,new_dict)
    return new_dict

def _merge_dicts_recursively(dict1, dict2, new_dict):
    for key in dict1:
        if type(dict1[key]) is dict:
            _merge_dicts_recursively(dict1[key],dict2[key], new_dict[key])
        elif type(dict1[key]) is np.ndarray:
            new_dict[key] = np.concatenate([dict1[key], dict2[key]])
        else:
            print('Unexpected type %s for key %s!' % (type(dict1[key]), key))

main_dict = merge_dicts(main_dict_2015, main_dict_2016)

if __name__ == '__main__':
    fontsz = 16

    plt.close('all')
    ms.mystyle_arial(fontsz=fontsz, dist_tick_lab=10)
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


    hl_keys = main_dict[moment]['heat_load'].keys()

    #sp_ctr = 0
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

    for fig in [fig1, fig2, fig3]:
        fig.suptitle('At '+moment)

    plt.show()
