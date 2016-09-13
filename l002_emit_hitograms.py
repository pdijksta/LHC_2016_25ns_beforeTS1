import LHCMeasurementTools.LHC_BCT as BCT
import LHCMeasurementTools.LHC_Energy as Energy
import LHCMeasurementTools.LHC_BSRT as BSRT
import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.mystyle as ms
import BSRT_calib 
import numpy as np
import pylab as pl
import pickle
import sys, time


filln = 5078

#~ filln = 5076
#~ filln = 5073
#~ filln = 5071
#~ filln = 5105
filln=5111
filln=5102
filln = 5112


fills_bmodes_file = 'fills_and_bmodes.pkl'
with open(fills_bmodes_file, 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)
    
if len(sys.argv)>1:
     print '--> Processing fill {:s}'.format(sys.argv[1])
     filln = int(sys.argv[1])

t_start_fill = dict_fill_bmodes[filln]['t_startfill']
t_end_fill = dict_fill_bmodes[filln]['t_endfill']
t_fill_len = t_end_fill - t_start_fill
t_ref = t_start_fill

na = np.array

bsrt_calib_dict = BSRT_calib.emittance_dictionary()

fill_dict = {}
fill_dict.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv'%filln, verbose=True))
fill_dict.update(tm.parse_timber_file('fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_%d.csv'%filln, verbose=True))
pl.close('all')
myfontsz = 14
ms.mystyle_arial(myfontsz) 
pl.figure(1, figsize=(8,12)).set_facecolor('w')
 
for beam_n in [1, 2]:
    energy = Energy.energy(fill_dict, beam=beam_n)
    bct = BCT.BCT(fill_dict, beam=beam_n)
    bsrt  = BSRT.BSRT(fill_dict, beam=beam_n, calib_dict=bsrt_calib_dict, average_repeated_meas=True)

    bsrt.calculate_emittances(energy)

    dict_bunches, t_bbb, emit_h_bbb, emit_v_bbb, bunch_n_un = bsrt.get_bbb_emit_evolution()

    dict_intervals = {}

    dict_intervals['Injection']={}
    dict_intervals['Injection']['t_start'] = dict_fill_bmodes[filln]['t_start_INJPHYS']
    dict_intervals['Injection']['t_end'] = dict_fill_bmodes[filln]['t_stop_PRERAMP']

    dict_intervals['he_before_SB']={}
    dict_intervals['he_before_SB']['t_start'] = dict_fill_bmodes[filln]['t_start_FLATTOP']
    dict_intervals['he_before_SB']['t_end'] = dict_fill_bmodes[filln]['t_start_STABLE']
    


    for interval in dict_intervals.keys():
        for kk in ['at_start', 'at_end', 'on_average']:
            dict_intervals[interval][kk] = {}
            dict_intervals[interval][kk]['emith'] = []
            dict_intervals[interval][kk]['emitv'] = []
            dict_intervals[interval][kk]['time_meas'] = []
            
    for slot_bun in bunch_n_un:
        for interval in dict_intervals.keys():
             
            mask_obs = np.logical_and(dict_bunches[slot_bun]['t_stamp']>dict_intervals[interval]['t_start'],
                                       dict_bunches[slot_bun]['t_stamp']<dict_intervals[interval]['t_end'])
                                       
            if np.sum(mask_obs)>0:
                dict_intervals[interval]['at_start']['emith'].append(dict_bunches[slot_bun]['norm_emit_h'][mask_obs][0])
                dict_intervals[interval]['at_start']['emitv'].append(dict_bunches[slot_bun]['norm_emit_v'][mask_obs][0])
                dict_intervals[interval]['at_start']['time_meas'].append(dict_bunches[slot_bun]['t_stamp'][mask_obs][0])
        
                dict_intervals[interval]['at_end']['emith'].append(dict_bunches[slot_bun]['norm_emit_h'][mask_obs][-1])
                dict_intervals[interval]['at_end']['emitv'].append(dict_bunches[slot_bun]['norm_emit_v'][mask_obs][-1])
                dict_intervals[interval]['at_end']['time_meas'].append(dict_bunches[slot_bun]['t_stamp'][mask_obs][-1])
                
                dict_intervals[interval]['on_average']['emith'].append(np.mean(dict_bunches[slot_bun]['norm_emit_h'][mask_obs]))
                dict_intervals[interval]['on_average']['emitv'].append(np.mean(dict_bunches[slot_bun]['norm_emit_v'][mask_obs]))
                dict_intervals[interval]['on_average']['time_meas'].append(np.mean(dict_bunches[slot_bun]['t_stamp'][mask_obs]))
            else:
                dict_intervals[interval]['at_start']['emith'].append(np.nan)
                dict_intervals[interval]['at_start']['emitv'].append(np.nan)
                dict_intervals[interval]['at_start']['time_meas'].append(np.nan)
        
                dict_intervals[interval]['at_end']['emith'].append(np.nan)
                dict_intervals[interval]['at_end']['emitv'].append(np.nan)
                dict_intervals[interval]['at_end']['time_meas'].append(np.nan)
                
                dict_intervals[interval]['on_average']['emith'].append(np.nan)
                dict_intervals[interval]['on_average']['emitv'].append(np.nan)
                dict_intervals[interval]['on_average']['time_meas'].append(np.nan)  
     
                
    n_bins = 50
    list_labels = ['Injected', 'Start Ramp', 'End Ramp', 'Start SB']
    
    for i_plane, plane in enumerate(['h', 'v']):
        pl.subplot(4,1,(beam_n-1)*2+i_plane+1)
        i_label = 0
        for interval in ['Injection', 'he_before_SB']:
            for moment in ['at_start', 'at_end']:
                print interval, moment, np.sum(dict_intervals[interval][moment]['emit'+plane])
                masknan = ~np.isnan(na(dict_intervals[interval][moment]['emit'+plane]))
                hist, bin_edges = np.histogram(na(dict_intervals[interval][moment]['emit'+plane])[masknan], range =(0,5), bins=n_bins)
                pl.step(bin_edges[:-1], hist, 
                label=list_labels[i_label]+', Avg. %.1f um'%np.mean(na(dict_intervals[interval][moment]['emit'+plane])[masknan]), 
                linewidth=2)
                i_label+=1
        pl.xlabel('Beam %d, Emittance %s'%(beam_n, plane))
        pl.ylabel('Occurrences')
        pl.grid('on')
        


    
for i_sp in xrange(4):    
    pl.subplot(4,1,i_sp+1)
    pl.legend(bbox_to_anchor=(1, 1),  loc='upper left', prop={'size':myfontsz})
pl.subplots_adjust(bottom=.05, hspace=.4, right=.55)
tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_ref))
pl.suptitle('Fill %d: started on %s'%(filln, tref_string), fontsize=18)
pl.savefig('emit_hist_plots/emihist_fill_%d.png'%filln, dpi=200)
pl.show()





