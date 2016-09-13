import LHCMeasurementTools.LHC_BCT as BCT
import LHCMeasurementTools.LHC_Energy as Energy
from LHCMeasurementTools.LHC_Stable_Phase import Phase, PowerLoss
import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.mystyle as ms
import numpy as np
import pylab as pl
import pickle
import sys, time
from colorsys import hsv_to_rgb
import os


colstr = {}
colstr[1] = 'b'
colstr[2] = 'r'

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

tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_ref))

N_traces_set = None

if len(sys.argv)>1:
	
     if np.any(map(lambda s: ('--n_traces'in s), sys.argv)):
        i_arg = np.where(map(lambda s: ('--n_traces'in s), sys.argv))[0]
        arg_temp = sys.argv[i_arg]
        N_traces_set = float(arg_temp.split('=')[-1])

	     
     if '--injection' in sys.argv:
		print 'Scans in the INJPHYS-PRERAMP beam modes'
		t_start_INJPHYS = dict_fill_bmodes[filln]['t_start_INJPHYS']
		t_start_RAMP = dict_fill_bmodes[filln]['t_start_RAMP']
		if N_traces_set==None: N_traces_set=30
		list_scan_times = np.linspace((t_start_INJPHYS-t_ref)/3600., (t_start_RAMP-t_ref)/3600., N_traces_set)

		
     if '--highenergy' in sys.argv:
		print 'Scans in the FLATTOP-STABLE beam modes'
		t_start_FLATTOP = dict_fill_bmodes[filln]['t_start_FLATTOP']
		t_start_STABLE = dict_fill_bmodes[filln]['t_start_STABLE']
		if N_traces_set==None: N_traces_set=30
		list_scan_times = np.linspace((t_start_FLATTOP-t_ref)/3600., (t_start_STABLE-t_ref)/3600.+0.5, N_traces_set)

     if '--stablebeams' in sys.argv:
		print 'Scans in the STABLE BEAMS'
		t_start_STABLE = dict_fill_bmodes[filln]['t_start_STABLE']
		t_end_STABLE = dict_fill_bmodes[filln]['t_stop_STABLE']
		if N_traces_set==None: N_traces_set=30
		list_scan_times = np.linspace((t_start_STABLE-t_ref)/3600., (t_end_STABLE-t_ref)/3600.+0.5, N_traces_set)
		
     if '--ramp' in sys.argv:
		print 'Scans in the RAMP'
		t_start_RAMP= dict_fill_bmodes[filln]['t_start_RAMP']
		t_end_RAMP = dict_fill_bmodes[filln]['t_stop_RAMP']
		if N_traces_set==None: N_traces_set=10
		list_scan_times = np.linspace((t_start_RAMP-t_ref)/3600., (t_end_RAMP-t_ref)/3600, N_traces_set)	
        
     if '--sigma' in sys.argv:
		plot_emittance=False
        
     if np.any(map(lambda s: ('--interval'in s), sys.argv)):
         i_arg = np.where(map(lambda s: ('--interval'in s), sys.argv))[0]
         arg_temp = sys.argv[i_arg]
         t_start_man = float(arg_temp.split('=')[-1].split(',')[0])
         t_end_man = float(arg_temp.split('=')[-1].split(',')[1])
         print 'Interval manually set: %.2fh to %.2fh'%(t_start_man, t_end_man)
         if N_traces_set==None: N_traces_set=30
         list_scan_times = np.linspace(t_start_man, t_end_man, N_traces_set)
         xlim = t_start_man, t_end_man
     else:
         xlim = None, None
        
     if '--notrace' in sys.argv:
        list_scan_times = []
              

fill_dict = {}
fill_dict.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv'%filln, verbose=True))


pl.close('all')
sp_ploss = None

# START PLOT
fig_h = pl.figure(1, figsize=(17,10))
fig_h.patch.set_facecolor('w')
ms.mystyle()
fig_h.suptitle('Fill %d: started on %s'%(filln, tref_string), fontsize=18)

sp_int = pl.subplot2grid((2,3), (0, 0), rowspan=1)
sp_energy = sp_int.twinx()

sp_totploss = pl.subplot2grid((2,3), (1, 0), rowspan=1, sharex=sp_int)

for beam in [1,2]:
    energy = Energy.energy(fill_dict, beam=beam)
    bct = BCT.BCT(fill_dict, beam=beam)
    

    ploss_filepath = 'Stable_phase_data/Power_Loss_Fill_%d_B%d.csv'%(filln, beam)
    if not os.path.isfile(ploss_filepath):
		print 'No file %s!'%ploss_filepath  
		ploss_bx = None
    else:
		ploss_bx = PowerLoss(ploss_filepath)

    # Intensity and energy

    sp_int.plot((bct.t_stamps - t_ref)/3600., bct.values, colstr[beam])
    mask_ene = energy.t_stamps > t_ref
    sp_energy.plot((energy.t_stamps[mask_ene] - t_ref)/3600., energy.energy[mask_ene]/1e3, 'k')
    
    sp_totploss.plot((ploss_bx.t_stamps - t_ref)/3600., ploss_bx.total_power_loss/1e3, colstr[beam])

    N_scans = len(list_scan_times)
    sp_ploss = pl.subplot2grid((2,3), (beam-1, 1), rowspan=1, colspan=2,  sharex=sp_ploss)
    sp_ploss.set_title('Beam %d'%beam)
    for ii in xrange(N_scans):
        colorcurr = hsv_to_rgb(float(ii)/float(N_scans), 0.9, 1.)
		
        t_curr = list_scan_times[ii]*3600. + t_ref
        
        ploss_curr, t_ploss_curr = ploss_bx.nearest_older_sample_power_loss(t_curr, flag_return_time=True)
        pl.plot(ploss_curr, '.', color = colorcurr, label='Dt=%.0fs'%(t_ploss_curr-t_curr))

        sp_int.axvline((t_ploss_curr - t_ref)/3600., color=colorcurr)
        sp_ploss.grid('on')
        sp_ploss.set_ylabel('Power loss [W]')
        sp_ploss.set_xlabel('Bunch slot')
        sp_ploss.set_xlim(0, 3500)
        sp_ploss.set_ylim(-5, 35)
    

sp_int.set_ylabel('Intensity [p$^+$]')
sp_energy.set_ylabel('Energy [TeV]')
sp_int.set_xlabel('Time [h]')
sp_int.grid('on')
sp_int.set_ylim(0, None)
sp_int.set_xlim(xlim)    
    
sp_totploss.set_ylabel('Total power loss [kW]')
sp_totploss.set_xlabel('Time [h]')
sp_totploss.set_ylim(-1, None)
sp_totploss.set_xlim(xlim)
sp_totploss.grid('on')

fig_h.subplots_adjust(top=0.9,right=0.95, left=0.07, hspace=0.3, wspace=0.45)	
fig_h.savefig('../for_evian_stable_phase/Stable_phase_%s.png'%(filln), dpi=220)

pl.show()
