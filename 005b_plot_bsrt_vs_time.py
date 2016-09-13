import LHCMeasurementTools.LHC_BCT as BCT
import LHCMeasurementTools.LHC_Energy as Energy
import LHCMeasurementTools.LHC_BSRT as BSRT
import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.mystyle as ms
import BSRT_calib 
import numpy as np
import pylab as pl
pl.switch_backend('TkAgg')
import pickle
import sys, time
from colorsys import hsv_to_rgb

# BSRT scan parameters
filln = 4914
list_scan_times = np.linspace(1.7, 2.25, 20)

filln = 4940
list_scan_times = np.linspace(1.7, 2.25, 20)

filln = 5085
list_scan_times = np.linspace(1.7, 2.25, 20)

scan_thrld = 100
plot_emittance = True


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
n_traces = 50.

bsrt_calib_dict = BSRT_calib.emittance_dictionary()

if len(sys.argv)>1:

     if '--sigma' in sys.argv:
                plot_emittance=False






fill_dict = {}
fill_dict.update(tm.parse_timber_file('fill_basic_data_csvs/basic_data_fill_%d.csv'%filln, verbose=False))
fill_dict.update(tm.parse_timber_file('fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_%d.csv'%filln, verbose=False))


pl.close('all')

beam_col = ['k', 'b','r']

sp_int = None
for beam in [1,2]:
    energy = Energy.energy(fill_dict, beam=beam)
    bct = BCT.BCT(fill_dict, beam=beam)
    bsrt  = BSRT.BSRT(fill_dict, beam=beam, calib_dict=bsrt_calib_dict)
    if plot_emittance:
        bsrt.calculate_emittances(energy)
        
    fig_h = pl.figure(100+beam, figsize=(8,10))
    fig_h.patch.set_facecolor('w')
    ms.mystyle()
    
    # Intensity and energy
    sp_int = pl.subplot2grid((3,1), (0, 0), rowspan=1, sharex=sp_int)
    sp_energy = sp_int.twinx()
    sp_int.grid('on')
    
    sp_int.plot((bct.t_stamps - t_ref)/3600., bct.values, 'b', linewidth=2, color=beam_col[beam])
    mask_ene = energy.t_stamps > t_ref
    sp_energy.plot((energy.t_stamps[mask_ene] - t_ref)/3600., energy.energy[mask_ene]/1e3, 'k', linewidth=2)
    
    sp_int.set_ylabel('Intensity [p$^+$]')
    sp_energy.set_ylabel('Energy [TeV]') 
    
    sp_emih = pl.subplot2grid((3,1), (1, 0), rowspan=1, sharex=sp_int)
    sp_emih.plot((bsrt.t_stamps - t_ref)/3600., bsrt.norm_emit_h, '.', markersize=.5, color=beam_col[beam])
    sp_emih.grid('on')
    sp_emih.set_ylabel('Emittance H [um]')
    sp_emih.set_ylim(0, None)
    
       
    sp_emiv = pl.subplot2grid((3,1), (2, 0), rowspan=1, sharex=sp_int)
    sp_emiv.plot((bsrt.t_stamps - t_ref)/3600., bsrt.norm_emit_v, '.', markersize=.5, color=beam_col[beam])    
    sp_emiv.grid('on')
    sp_emiv.set_ylabel('Emittance V [um]')
    sp_emiv.set_ylim(0, None)
    sp_emiv.set_xlabel('Time [h]')
    
    
    tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_ref))
    fig_h.suptitle('Fill %d: B%d, started on %s'%(filln, bsrt.beam, tref_string), fontsize=18)
    fig_h.subplots_adjust(top=0.9,right=0.88, left=0.12, hspace=0.3, wspace=0.45)


pl.show()
