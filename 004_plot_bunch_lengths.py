import LHCMeasurementTools.mystyle as ms
import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.TimestampHelpers as th
import LHCMeasurementTools.LHC_BCT as BCT
import LHCMeasurementTools.LHC_BQM as BQM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mlc
import matplotlib.dates as mdt
import pickle, sys, time, string

pkl_name = 'fills_and_bmodes.pkl'
with open(pkl_name, 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)

if len(sys.argv)>1:
     print '--> Processing fill {:s}'.format(sys.argv[1])
     filln = int(sys.argv[1])

t_ref = dict_fill_bmodes[filln]['t_startfill']
t_end = dict_fill_bmodes[filln]['t_endfill']
tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_ref))

traces_times = np.linspace(0.1, (t_end-t_ref)/3600., 20)
i_bun_obs_list = np.arange(0,500)[::2]
# t_loss_ref = 1.97

N_traces_set = None

if len(sys.argv)>1:
    
     if np.any(map(lambda s: ('--n_traces'in s), sys.argv)):
        i_arg = np.where(map(lambda s: ('--n_traces'in s), sys.argv))[0]
        arg_temp = sys.argv[i_arg]
        N_traces_set = float(arg_temp.split('=')[-1])
        traces_times = np.linspace(0.1, (t_end-t_ref)/3600., N_traces_set)
         
     if '--injection' in sys.argv:
        print 'Scans in the INJPHYS-PRERAMP beam modes'
        t_start_INJPHYS = dict_fill_bmodes[filln]['t_start_INJPHYS']
        t_start_RAMP = dict_fill_bmodes[filln]['t_start_RAMP']
        if N_traces_set==None: N_traces_set=30
        traces_times = np.linspace((t_start_INJPHYS-t_ref)/3600., (t_start_RAMP-t_ref)/3600., N_traces_set)

        
     if '--highenergy' in sys.argv:
        print 'Scans in the FLATTOP-STABLE beam modes'
        t_start_FLATTOP = dict_fill_bmodes[filln]['t_start_FLATTOP']
        t_start_STABLE = dict_fill_bmodes[filln]['t_start_STABLE']
        if N_traces_set==None: N_traces_set=30
        traces_times = np.linspace((t_start_FLATTOP-t_ref)/3600., (t_start_STABLE-t_ref)/3600.+0.5, N_traces_set)

     if '--stablebeams' in sys.argv:
        print 'Scans in the STABLE BEAMS'
        t_start_STABLE = dict_fill_bmodes[filln]['t_start_STABLE']
        t_end_STABLE = dict_fill_bmodes[filln]['t_stop_STABLE']
        if N_traces_set==None: N_traces_set=30
        traces_times = np.linspace((t_start_STABLE-t_ref)/3600., (t_end_STABLE-t_ref)/3600.+0.5, N_traces_set)
        
     if '--ramp' in sys.argv:
        print 'Scans in the RAMP'
        t_start_RAMP= dict_fill_bmodes[filln]['t_start_RAMP']
        t_end_RAMP = dict_fill_bmodes[filln]['t_stop_RAMP']
        if N_traces_set==None: N_traces_set=10
        traces_times = np.linspace((t_start_RAMP-t_ref)/3600., (t_end_RAMP-t_ref)/3600, N_traces_set)    

        
     if np.any(map(lambda s: ('--interval'in s), sys.argv)):
        i_arg = np.where(map(lambda s: ('--interval'in s), sys.argv))[0]
        arg_temp = sys.argv[i_arg]
        t_start_man = float(arg_temp.split('=')[-1].split(',')[0])
        t_end_man = float(arg_temp.split('=')[-1].split(',')[1])
        print 'Interval manually set: %.2fh to %.2fh'%(t_start_man, t_end_man)
        if N_traces_set==None: N_traces_set=30
        traces_times = np.linspace(t_start_man, t_end_man, N_traces_set)
        
     if '--notrace' in sys.argv:
        traces_times = []



plt.rcParams.update({'axes.labelsize': 18,
                     'axes.linewidth': 2,
                     'xtick.labelsize': 'large',
                     'ytick.labelsize': 'large',
                     'xtick.major.pad': 14,
                     'ytick.major.pad': 14})

format_datetime = mdt.DateFormatter('%m-%d %H:%M')


fill_dict = {}
fill_dict.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv'%filln, verbose=False))
fill_dict.update(tm.parse_timber_file('fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_%d.csv'%filln, verbose=False))

n_traces = len(traces_times)
blen_thresh = 0.



i_fig = 0
plt.close('all')
beam_col = ['b','r']
for beam in [1,2]:
    blength = BQM.blength(fill_dict, beam = beam)
    bct = BCT.BCT(fill_dict, beam=beam)
    
    fig1 = plt.figure(i_fig, figsize=(14, 8), tight_layout=False)
    fig1.patch.set_facecolor('w')
    ax0 = plt.subplot(211)
    ax1 = plt.subplot(212)
    ms.mystyle()

    ax0.plot((bct.t_stamps-t_ref)/3600., bct.values, color=beam_col[beam-1], lw=2)
    for i in xrange(0, n_traces):
        t_cut_h = traces_times[i]
        t_curr = t_ref+t_cut_h*3600.	
        blength_curr, t_blength_curr = blength.nearest_older_sample(t_curr, flag_return_time=True)
            
        ax1.plot(blength_curr, '.', color=ms.colorprog(i, n_traces),
                     label='%.2f h'%((t_blength_curr-t_ref)/3600.))
        ax0.axvline((t_blength_curr-t_ref)/3600., lw=1.5, color=ms.colorprog(i, n_traces))

    ax1.set_xlabel('25 ns slot')
    ax1.set_xlim(0, 3500)
    ax1.set_ylabel('Bunch length [s]')
    ax1.grid('on')
    ax0.set_xlabel('Time [h]')
    ax0.grid('on')
    ax0.set_ylim(bottom=.5e-9)
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.95, 
                        left=0.1, hspace=0.3, wspace=0.4)
    fig1.suptitle('Fill %d: B%d, started on %s'%(filln, beam, tref_string), fontsize=20)
    i_fig += 1
    
    #find the filled slots
    mask_bunches = np.ma.masked_greater(blength.blen, blen_thresh).mask
    list_nbunches = np.sum(mask_bunches, axis=0)
    i_bunches = np.where(list_nbunches > 0)[0]
    
    N_bunches = len(i_bunches)
    figbbb = plt.figure(100+i_fig, figsize=(14, 8), tight_layout=False)
    figbbb.patch.set_facecolor('w')
    axbbb_bct = plt.subplot(211)
    axbbb_traces = plt.subplot(212, sharex=axbbb_bct)
    
    #figlosref = plt.figure(200+i_fig, figsize=(14, 8), tight_layout=False)
    #figlosref.patch.set_facecolor('w')
    #axlref_bct = plt.subplot(211)
    #axlref_traces = plt.subplot(212, sharex=axbbb_bct)
    
    i_bunches_sel = list(set(i_bunches) & set(i_bun_obs_list))
    
    #totint_at_loss_ref = bct.nearest_older_sample(t_ref+t_loss_ref*3600)
    
    axbbb_bct.plot((bct.t_stamps-t_ref)/3600., bct.values, color=beam_col[beam-1], lw=2)
    for i_line, i_bun in enumerate(i_bunches_sel):
        axbbb_traces.plot((blength.t_stamps-t_ref)/3600., blength.blen[:, i_bun],
        color=ms.colorprog(i_line, len(i_bunches_sel)))
    axbbb_bct.grid('on')
    axbbb_traces.grid('on')

plt.show()
