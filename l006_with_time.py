import LHCMeasurementTools.LHC_BCT as BCT
import LHCMeasurementTools.LHC_FBCT as FBCT
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
filln = 5073
#~ filln = 5071
#~ filln = 5105
#~ filln=5111
#~ filln=5102
#~ filln = 5112
filln=5161


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
myfontsz = 11
ms.mystyle_arial(myfontsz) 

#preapare empty figure for histograms
fig_emit_hist = pl.figure(1, figsize=(14,8))
fig_emit_hist.set_facecolor('w')
sp_emit_hist_list = []
sptemp = None
for i_sp in xrange(4):
    sptemp = pl.subplot(4,2,i_sp*2+1, sharex = sptemp)
    sp_emit_hist_list.append(sptemp)
sp_inten_hist_list = []
sptemp = None
for i_sp in xrange(2):
    sptemp = pl.subplot(4,2,i_sp*2+2, sharex = sptemp)
    sp_inten_hist_list.append(sptemp) 
sp_bright_hist_list = []
sptemp = None
for i_sp in xrange(2):
    sptemp = pl.subplot(4,2,i_sp*2+6, sharex = sptemp)
    sp_bright_hist_list.append(sptemp) 
    
#preapare empty figure bunch by bunch emit
fig_emit_bbb = pl.figure(2, figsize=(14, 8))
fig_emit_bbb.set_facecolor('w')
sp_emit_bbb_list = []
sptemp = None
for i_sp in xrange(4):
    sptemp = pl.subplot(4,1,i_sp+1, sharex=sptemp, sharey=sptemp)
    sp_emit_bbb_list.append(sptemp)
    
#preapare empty figure bunch by bunch intensity
fig_inten_bbb = pl.figure(3, figsize=(14, 8))
fig_inten_bbb.set_facecolor('w')
sp_inten_bbb_list = []
sptemp = None
for i_sp in xrange(2):
    sptemp = pl.subplot(2,1,i_sp+1, sharex=sptemp, sharey=sptemp)
    sp_inten_bbb_list.append(sptemp)
    
#preapare empty figure bunch by bunch brightness
fig_bright_bbb = pl.figure(4, figsize=(14, 8))
fig_bright_bbb.set_facecolor('w')
sp_bright_bbb_list = []
sptemp = None
for i_sp in xrange(2):
    sptemp = pl.subplot(2,1,i_sp+1, sharex=sptemp, sharey=sptemp)
    sp_bright_bbb_list.append(sptemp)
    
#preapare empty figure bunch by bunch time
fig_time_bbb = pl.figure(5, figsize=(14, 8))
fig_time_bbb.set_facecolor('w')
sp_time_bbb_list = []
sptemp = None
for i_sp in xrange(2):
    sptemp = pl.subplot(2,1,i_sp+1, sharex=sptemp, sharey=sptemp)
    sp_time_bbb_list.append(sptemp)

#preapare empty figure bunch by bunch time
fig_time_bbb = pl.figure(5, figsize=(14, 8))
fig_time_bbb.set_facecolor('w')
sp_time_bbb_list = []
sptemp = None
for i_sp in xrange(2):
    sptemp = pl.subplot(2,1,i_sp+1, sharex=sptemp, sharey=sptemp)
    sp_time_bbb_list.append(sptemp)
 
for beam_n in [1, 2]:
    energy = Energy.energy(fill_dict, beam=beam_n)
    bct = BCT.BCT(fill_dict, beam=beam_n)
    bsrt  = BSRT.BSRT(fill_dict, beam=beam_n, calib_dict=bsrt_calib_dict, average_repeated_meas=True)
    fbct = FBCT.FBCT(fill_dict, beam=beam_n)
    
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
        for kk in ['at_start', 'at_end']:#, 'on_average']:
            dict_intervals[interval][kk] = {}
            dict_intervals[interval][kk]['emith'] = []
            dict_intervals[interval][kk]['emitv'] = []
            dict_intervals[interval][kk]['time_meas'] = []
            dict_intervals[interval][kk]['intensity'] = []
            
    for slot_bun in bunch_n_un:
        for interval in dict_intervals.keys():
            dict_intervals[interval]['filled_slots'] = bunch_n_un
             
            mask_obs = np.logical_and(dict_bunches[slot_bun]['t_stamp']>dict_intervals[interval]['t_start'],
                                       dict_bunches[slot_bun]['t_stamp']<dict_intervals[interval]['t_end'])
                                       
                                       
            if np.sum(mask_obs)>0:
                
                dict_intervals[interval]['at_start']['emith'].append(dict_bunches[slot_bun]['norm_emit_h'][mask_obs][0])
                dict_intervals[interval]['at_start']['emitv'].append(dict_bunches[slot_bun]['norm_emit_v'][mask_obs][0])
                dict_intervals[interval]['at_start']['time_meas'].append(dict_bunches[slot_bun]['t_stamp'][mask_obs][0])
                dict_intervals[interval]['at_start']['intensity'].append(
                    np.interp(dict_intervals[interval]['at_start']['time_meas'][-1]+100.,
                                             fbct.t_stamps, fbct.bint[:, slot_bun]))
                    
                
        
                dict_intervals[interval]['at_end']['emith'].append(dict_bunches[slot_bun]['norm_emit_h'][mask_obs][-1])
                dict_intervals[interval]['at_end']['emitv'].append(dict_bunches[slot_bun]['norm_emit_v'][mask_obs][-1])
                dict_intervals[interval]['at_end']['time_meas'].append(dict_bunches[slot_bun]['t_stamp'][mask_obs][-1])
                dict_intervals[interval]['at_end']['intensity'].append(
                    np.interp(dict_intervals[interval]['at_end']['time_meas'][-1],
                                             fbct.t_stamps, fbct.bint[:, slot_bun]))
                
                #~ dict_intervals[interval]['on_average']['emith'].append(np.mean(dict_bunches[slot_bun]['norm_emit_h'][mask_obs]))
                #~ dict_intervals[interval]['on_average']['emitv'].append(np.mean(dict_bunches[slot_bun]['norm_emit_v'][mask_obs]))
                #~ dict_intervals[interval]['on_average']['time_meas'].append(np.mean(dict_bunches[slot_bun]['t_stamp'][mask_obs]))

                
            else:
                
                dict_intervals[interval]['at_start']['emith'].append(np.nan)
                dict_intervals[interval]['at_start']['emitv'].append(np.nan)
                dict_intervals[interval]['at_start']['time_meas'].append(np.nan)
                dict_intervals[interval]['at_start']['intensity'].append(np.nan)
        
                dict_intervals[interval]['at_end']['emith'].append(np.nan)
                dict_intervals[interval]['at_end']['emitv'].append(np.nan)
                dict_intervals[interval]['at_end']['time_meas'].append(np.nan)
                dict_intervals[interval]['at_end']['intensity'].append(np.nan)        
                        
                #~ dict_intervals[interval]['on_average']['emith'].append(np.nan)
                #~ dict_intervals[interval]['on_average']['emitv'].append(np.nan)
                #~ dict_intervals[interval]['on_average']['time_meas'].append(np.nan)  
                #~ dict_intervals[interval]['on_average']['intensity'].append(np.nan)        

     
                
    n_bins_emit = 50
    list_labels = ['Injected', 'Start Ramp', 'End Ramp', 'Start SB']
    
    # emittance plots
    for i_plane, plane in enumerate(['h', 'v']):
        i_sp = (beam_n-1)*2+i_plane
        i_label = 0
        for interval in ['Injection', 'he_before_SB']:
            for moment in ['at_start', 'at_end']:
                masknan = ~np.isnan(na(dict_intervals[interval][moment]['emit'+plane]))
                print interval, moment, np.sum(dict_intervals[interval][moment]['emit'+plane]), np.sum(~masknan)
                hist, bin_edges = np.histogram(na(dict_intervals[interval][moment]['emit'+plane])[masknan], range =(0,5), bins=n_bins_emit)
                
                sp_emit_hist_list[i_sp].step(bin_edges[:-1], hist, 
                                    label=list_labels[i_label]+', Avg. %.1f um'%np.mean(na(dict_intervals[interval][moment]['emit'+plane])[masknan]), 
                                    linewidth=1)
                sp_emit_bbb_list[i_sp].plot(na(bunch_n_un)[masknan], na(dict_intervals[interval][moment]['emit'+plane])[masknan], '.',
                                    label=list_labels[i_label]+', Avg. %.1f um'%np.mean(na(dict_intervals[interval][moment]['emit'+plane])[masknan]))
                
                i_label+=1
        sp_emit_hist_list[i_sp].set_xlabel('Beam %d, Emittance %s [um]'%(beam_n, plane))
        sp_emit_hist_list[i_sp].set_ylabel('Occurrences')
        sp_emit_hist_list[i_sp].grid('on')
        sp_emit_hist_list[i_sp].tick_params(axis='both', which='major', pad=5)
        sp_emit_hist_list[i_sp].xaxis.labelpad = 1
        sp_emit_hist_list[i_sp].ticklabel_format(style='sci', scilimits=(0,0),axis='y') 

        
        sp_emit_bbb_list[i_sp].set_ylabel('B%d, Emitt. %s [um]'%(beam_n, plane))
        sp_emit_bbb_list[i_sp].grid('on')
        sp_emit_bbb_list[i_sp].tick_params(axis='both', which='major', pad=5)
        sp_emit_bbb_list[i_sp].xaxis.labelpad = 1
        
        
    #intensity plots
    i_label = 0
    i_sp = beam_n-1
    n_bins_inten = 50
    for interval in ['Injection', 'he_before_SB']:
            for moment in ['at_start', 'at_end']:
                sp_inten_bbb_list[i_sp].plot(na(bunch_n_un)[masknan], na(dict_intervals[interval][moment]['intensity'])[masknan], '.',
                                    label=list_labels[i_label]+', Avg. %.2fe11'%(np.mean(na(dict_intervals[interval][moment]['intensity'])[masknan])/1e11))
                hist, bin_edges = np.histogram(na(dict_intervals[interval][moment]['intensity'])[masknan], range =(0.5e11,1.5e11), bins=n_bins_inten)
                sp_inten_hist_list[i_sp].step(bin_edges[:-1], hist, 
                                    label=list_labels[i_label]+', Avg. %.2fe11'%(np.mean(na(dict_intervals[interval][moment]['intensity'])[masknan])/1e11), 
                                    linewidth=1)
                i_label+=1
                
    sp_inten_bbb_list[i_sp].set_ylabel('Beam %d, Intensity [p/b]'%(beam_n))
    sp_inten_bbb_list[i_sp].grid('on')
    sp_inten_bbb_list[i_sp].tick_params(axis='both', which='major', pad=5)
    sp_inten_bbb_list[i_sp].xaxis.labelpad = 1
    sp_inten_bbb_list[i_sp].ticklabel_format(style='sci', scilimits=(0,0),axis='y') 
    
    sp_inten_hist_list[i_sp].set_xlabel('Beam %d, Intensity [p/b]'%(beam_n))
    sp_inten_hist_list[i_sp].set_ylabel('Occurrences')
    sp_inten_hist_list[i_sp].grid('on')
    sp_inten_hist_list[i_sp].tick_params(axis='both', which='major', pad=5)
    sp_inten_hist_list[i_sp].xaxis.labelpad = 1
    sp_inten_hist_list[i_sp].ticklabel_format(style='sci', scilimits=(0,0),axis='y') 
    

    # compute brightness
    for interval in ['Injection', 'he_before_SB']:
            for moment in ['at_start', 'at_end']:
                dict_intervals[interval][moment]['brightness'] = \
                    na(dict_intervals[interval][moment]['intensity'])/np.sqrt(na(dict_intervals[interval][moment]['emith'])*na(dict_intervals[interval][moment]['emitv']))
    # plot brightness
    i_label = 0
    i_sp = beam_n-1
    for interval in ['Injection', 'he_before_SB']:
        for moment in ['at_start', 'at_end']:
                sp_bright_bbb_list[i_sp].plot(na(bunch_n_un)[masknan], na(dict_intervals[interval][moment]['brightness'])[masknan], '.',
                    label=list_labels[i_label]+', Avg. %.2fe11'%(np.mean(na(dict_intervals[interval][moment]['brightness'])[masknan])/1e11))
                hist, bin_edges = np.histogram(na(dict_intervals[interval][moment]['brightness'])[masknan], range =(0,1e11), bins=n_bins_inten)
                sp_bright_hist_list[i_sp].step(bin_edges[:-1], hist, 
                                    label=list_labels[i_label]+', Avg. %.2fe11'%(np.mean(na(dict_intervals[interval][moment]['brightness'])[masknan])/1e11), 
                                    linewidth=1)
                i_label+=1
                
    sp_bright_bbb_list[i_sp].set_ylabel('Beam %d, Brightness [p/um/b]'%(beam_n))
    sp_bright_bbb_list[i_sp].grid('on')
    sp_bright_bbb_list[i_sp].tick_params(axis='both', which='major', pad=5)
    sp_bright_bbb_list[i_sp].xaxis.labelpad = 1
    sp_bright_bbb_list[i_sp].ticklabel_format(style='sci', scilimits=(0,0),axis='y')
    
    sp_bright_hist_list[i_sp].set_xlabel('Beam %d, Brightness [p/um]'%(beam_n))
    sp_bright_hist_list[i_sp].set_ylabel('Occurrences')
    sp_bright_hist_list[i_sp].grid('on')
    sp_bright_hist_list[i_sp].tick_params(axis='both', which='major', pad=5)
    sp_bright_hist_list[i_sp].xaxis.labelpad = 1
    sp_bright_hist_list[i_sp].ticklabel_format(style='sci', scilimits=(0,0),axis='y')
    
    # plot time
    i_label = 0
    i_sp = beam_n-1
    for interval in ['Injection', 'he_before_SB']:
                sp_time_bbb_list[i_sp].plot(na(bunch_n_un)[masknan], (na(dict_intervals[interval]['at_end']['time_meas'])[masknan]-\
                                                                       na(dict_intervals[interval]['at_start']['time_meas'])[masknan])/60. , '.',
                                    label=', Avg. %.2f'%(np.mean(na(dict_intervals[interval]['at_end']['time_meas'])[masknan]-\
                                                                       na(dict_intervals[interval]['at_start']['time_meas'])[masknan])/60.))
         
    sp_time_bbb_list[i_sp].set_ylabel('Beam %d, Time start to end [min]'%(beam_n))
    sp_time_bbb_list[i_sp].grid('on')
    sp_time_bbb_list[i_sp].tick_params(axis='both', which='major', pad=5)
    sp_time_bbb_list[i_sp].xaxis.labelpad = 1             


sp_emit_bbb_list[-1].set_xlabel('25 ns slot')    
sp_emit_bbb_list[-1].set_ylim(1., 5.)

sp_inten_bbb_list[-1].set_xlabel('25 ns slot')
 
for sp in sp_emit_hist_list:    
    sp.legend(bbox_to_anchor=(1, 1),  loc='upper left', prop={'size':myfontsz})
    
for sp in sp_emit_bbb_list:    
    sp.legend(bbox_to_anchor=(1, 1),  loc='upper left', prop={'size':myfontsz})
    
for sp in sp_inten_bbb_list:    
    sp.legend(bbox_to_anchor=(1, 1),  loc='upper left', prop={'size':myfontsz})    
    
for sp in sp_bright_bbb_list:    
    sp.legend(bbox_to_anchor=(1, 1),  loc='upper left', prop={'size':myfontsz})  
    
for sp in sp_inten_hist_list:    
    sp.legend(bbox_to_anchor=(1, 1),  loc='upper left', prop={'size':myfontsz})
    
for sp in sp_bright_hist_list:    
    sp.legend(bbox_to_anchor=(1, 1),  loc='upper left', prop={'size':myfontsz}) 

for sp in sp_time_bbb_list:
    sp.legend(bbox_to_anchor=(1, 1),  loc='upper left', prop={'size':myfontsz})     
    
fig_emit_hist.subplots_adjust(left=.09, bottom=.07, right=.76, top=.92, wspace=1., hspace=.55)
tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_ref))

fig_emit_hist.suptitle('Fill %d: started on %s'%(filln, tref_string), fontsize=myfontsz)
fig_emit_hist.savefig('emit_hist_plots/emihist_fill_%d.png'%filln, dpi=200)

fig_emit_bbb.subplots_adjust(left=.05, right=.81, top=.93)
fig_emit_bbb.suptitle('Fill %d: started on %s'%(filln, tref_string), fontsize=myfontsz)
fig_emit_bbb.savefig('emit_hist_plots/emibbb_fill_%d.png'%filln, dpi=200)

fig_inten_bbb.subplots_adjust(left=.05, right=.81, top=.93)
fig_inten_bbb.suptitle('Fill %d: started on %s'%(filln, tref_string), fontsize=myfontsz)
fig_inten_bbb.savefig('emit_hist_plots/intenbbb_fill_%d.png'%filln, dpi=200)

fig_bright_bbb.subplots_adjust(left=.05, right=.81, top=.93)
fig_bright_bbb.suptitle('Fill %d: started on %s'%(filln, tref_string), fontsize=myfontsz)

fig_time_bbb.subplots_adjust(left=.05, right=.81, top=.93)
fig_time_bbb.suptitle('Fill %d: started on %s'%(filln, tref_string), fontsize=myfontsz)

pl.show()





